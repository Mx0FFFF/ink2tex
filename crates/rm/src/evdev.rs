//! Hand-rolled evdev / ioctl layer — the Linux input subsystem, up close.
//!
//! ## What this exercises (systems concept: character devices + ioctl capability probing)
//! The kernel exposes every input device as a character device at
//! `/dev/input/eventN`. Two ways to talk to it:
//!   * `read(2)` returns a stream of fixed-size `struct input_event` records — the
//!     actual pen samples.
//!   * `ioctl(2)` with the `EVIOC*` request codes *interrogates* the device:
//!     its name (`EVIOCGNAME`), which event types/codes it can emit
//!     (`EVIOCGBIT`), and the range of each absolute axis (`EVIOCGABS`).
//!
//! We do this by hand (raw `libc`, no `evdev` crate) because the point of the
//! project is to understand the syscall layer. An ioctl request number is not a
//! magic constant — it *encodes* direction, a type byte, a command number, and
//! the size of the argument struct. `ioc()` below builds it exactly as the C
//! macro `_IOC` does.
//!
//! We never hardcode `event1`: numbering isn't stable across firmware. We
//! enumerate and keep the device that advertises `BTN_TOOL_PEN` + `ABS_PRESSURE`
//! (see `docs/device.md`).

use std::ffi::{CStr, CString};
use std::io;
use std::os::unix::io::RawFd;

// ---- Linux ioctl request encoding (asm-generic; ARM uses this variant) -------
// A request packs: [dir:2][size:14][type:8][nr:8]. `_IOC` in <asm-generic/ioctl.h>.
const IOC_NRSHIFT: u32 = 0;
const IOC_TYPESHIFT: u32 = 8; // after 8 nr bits
const IOC_SIZESHIFT: u32 = 16; // after 8 type bits
const IOC_DIRSHIFT: u32 = 30; // after 14 size bits
const IOC_READ: u32 = 2; // "device writes back into our buffer"
const EVDEV_TYPE: u32 = b'E' as u32;

/// Build an ioctl request number, mirroring the C `_IOC(dir,type,nr,size)` macro.
const fn ioc(dir: u32, nr: u32, size: u32) -> libc::c_ulong {
    ((dir << IOC_DIRSHIFT)
        | (EVDEV_TYPE << IOC_TYPESHIFT)
        | (nr << IOC_NRSHIFT)
        | (size << IOC_SIZESHIFT)) as libc::c_ulong
}

// EVIOCGNAME(len) = _IOC(READ,'E',0x06,len); EVIOCGBIT(ev,len)=_IOC(READ,'E',0x20+ev,len);
// EVIOCGABS(abs)  = _IOR('E',0x40+abs, struct input_absinfo).
fn eviocgname(len: u32) -> libc::c_ulong {
    ioc(IOC_READ, 0x06, len)
}
fn eviocgbit(ev: u32, len: u32) -> libc::c_ulong {
    ioc(IOC_READ, 0x20 + ev, len)
}
fn eviocgabs(abs: u32) -> libc::c_ulong {
    ioc(IOC_READ, 0x40 + abs, std::mem::size_of::<AbsInfo>() as u32)
}

// ---- evdev protocol constants we care about ---------------------------------
pub const EV_SYN: u16 = 0x00;
pub const EV_KEY: u16 = 0x01;
pub const EV_ABS: u16 = 0x03;

pub const SYN_REPORT: u16 = 0x00; // "one full sample is complete"

pub const BTN_TOOL_PEN: u16 = 0x140; // pen entered detection range
pub const BTN_TOUCH: u16 = 0x14a; // pen tip actually pressed to glass

pub const ABS_X: u16 = 0x00;
pub const ABS_Y: u16 = 0x01;
pub const ABS_PRESSURE: u16 = 0x18;
pub const ABS_DISTANCE: u16 = 0x19;
pub const ABS_TILT_X: u16 = 0x1a;
pub const ABS_TILT_Y: u16 = 0x1b;

const KEY_MAX: usize = 0x2ff;
const ABS_MAX: usize = 0x3f;

/// Kernel `struct input_event` for a 32-bit kernel (reMarkable is armv7l): a
/// 2×32-bit `timeval`, then type/code/value. `read()` returns whole multiples of
/// this. We derive `Copy` so we can read directly into an aligned array.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InputEvent {
    pub tv_sec: libc::time_t,
    pub tv_usec: libc::suseconds_t,
    pub kind: u16, // `type` in C; reserved word here
    pub code: u16,
    pub value: i32,
}

impl InputEvent {
    /// A zeroed event, for allocating a read buffer: `[InputEvent::zeroed(); N]`.
    pub const fn zeroed() -> Self {
        Self {
            tv_sec: 0,
            tv_usec: 0,
            kind: 0,
            code: 0,
            value: 0,
        }
    }

    /// Microseconds encoded by this event's timestamp (the kernel stamps each
    /// sample with `CLOCK_REALTIME`; we only ever use *differences*, so the epoch
    /// doesn't matter — it becomes `Point::t_us`).
    // `as i64` is a no-op on the 64-bit host but a real widening on the 32-bit
    // device (time_t/suseconds_t are i32 there); keep it for the target.
    #[allow(clippy::unnecessary_cast)]
    pub fn t_us(&self) -> u64 {
        (self.tv_sec as i64 as u64).wrapping_mul(1_000_000) + (self.tv_usec as i64 as u64)
    }
}

/// Kernel `struct input_absinfo` — the range metadata for one absolute axis.
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct AbsInfo {
    pub value: i32,
    pub minimum: i32,
    pub maximum: i32,
    pub fuzz: i32,
    pub flat: i32,
    pub resolution: i32,
}

/// RAII file descriptor — closes on drop so no path leaks an fd (no panics).
pub struct Fd(RawFd);

impl Fd {
    pub fn raw(&self) -> RawFd {
        self.0
    }
}

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: we own this fd for our lifetime; close is the only call.
        unsafe { libc::close(self.0) };
    }
}

fn open_ro(path: &CStr) -> io::Result<Fd> {
    // SAFETY: path is a valid NUL-terminated C string; O_RDONLY read-only open.
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_RDONLY) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Fd(fd))
}

/// `ioctl` with a pointer argument, surfacing errno as an `io::Error`.
///
/// # Safety
/// `buf` must point to at least the number of bytes encoded in `req`'s size field.
unsafe fn ioctl(fd: RawFd, req: libc::c_ulong, buf: *mut libc::c_void) -> io::Result<i32> {
    let r = libc::ioctl(fd, req, buf);
    if r < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(r)
    }
}

fn test_bit(bytes: &[u8], bit: usize) -> bool {
    bytes
        .get(bit / 8)
        .is_some_and(|b| b & (1 << (bit % 8)) != 0)
}

fn get_name(fd: RawFd) -> io::Result<String> {
    let mut buf = [0u8; 256];
    // SAFETY: buf is 256 bytes, matching the len encoded in the request.
    unsafe { ioctl(fd, eviocgname(buf.len() as u32), buf.as_mut_ptr() as *mut _)? };
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
}

fn get_abs(fd: RawFd, axis: u16) -> io::Result<AbsInfo> {
    let mut info = AbsInfo::default();
    // SAFETY: &mut info is exactly size_of::<AbsInfo>() bytes, matching the request.
    unsafe {
        ioctl(
            fd,
            eviocgabs(axis as u32),
            &mut info as *mut AbsInfo as *mut _,
        )?
    };
    Ok(info)
}

/// A digitizer we found and probed. Ranges are read live via `EVIOCGABS` — no
/// hardcoded lore (`docs/device.md` rule).
pub struct Digitizer {
    pub path: String,
    pub name: String,
    pub fd: Fd,
    pub x: AbsInfo,
    pub y: AbsInfo,
    pub pressure: AbsInfo,
    pub tilt_x: AbsInfo,
    pub tilt_y: AbsInfo,
    pub distance: AbsInfo,
}

/// Does this fd advertise the pen? A digitizer emits `BTN_TOOL_PEN` (an EV_KEY)
/// and `ABS_PRESSURE` (an EV_ABS). We ask the kernel via `EVIOCGBIT`.
fn is_digitizer(fd: RawFd) -> io::Result<bool> {
    let mut ev = [0u8; 4]; // EV_MAX/8 + 1
                           // SAFETY: 4-byte buffer matches the requested length.
    unsafe { ioctl(fd, eviocgbit(0, ev.len() as u32), ev.as_mut_ptr() as *mut _)? };
    if !test_bit(&ev, EV_KEY as usize) || !test_bit(&ev, EV_ABS as usize) {
        return Ok(false);
    }
    let mut keys = [0u8; KEY_MAX / 8 + 1];
    // SAFETY: buffer sized to hold all key bits.
    unsafe {
        ioctl(
            fd,
            eviocgbit(EV_KEY as u32, keys.len() as u32),
            keys.as_mut_ptr() as *mut _,
        )?
    };
    let mut abs = [0u8; ABS_MAX / 8 + 1];
    // SAFETY: buffer sized to hold all abs bits.
    unsafe {
        ioctl(
            fd,
            eviocgbit(EV_ABS as u32, abs.len() as u32),
            abs.as_mut_ptr() as *mut _,
        )?
    };
    Ok(test_bit(&keys, BTN_TOOL_PEN as usize) && test_bit(&abs, ABS_PRESSURE as usize))
}

/// Enumerate `/dev/input/event*`, probe each, return the first pen digitizer.
pub fn find_digitizer() -> io::Result<Digitizer> {
    let mut nodes: Vec<String> = std::fs::read_dir("/dev/input")?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("event"))
        })
        .filter_map(|p| p.to_str().map(str::to_owned))
        .collect();
    nodes.sort(); // deterministic order (event0, event1, ...)

    for path in nodes {
        let cpath = match CString::new(path.as_str()) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let fd = match open_ro(&cpath) {
            Ok(fd) => fd,
            Err(_) => continue, // some nodes need privileges we may lack; skip
        };
        if is_digitizer(fd.raw()).unwrap_or(false) {
            let raw = fd.raw();
            return Ok(Digitizer {
                name: get_name(raw).unwrap_or_default(),
                x: get_abs(raw, ABS_X)?,
                y: get_abs(raw, ABS_Y)?,
                pressure: get_abs(raw, ABS_PRESSURE)?,
                tilt_x: get_abs(raw, ABS_TILT_X).unwrap_or_default(),
                tilt_y: get_abs(raw, ABS_TILT_Y).unwrap_or_default(),
                distance: get_abs(raw, ABS_DISTANCE).unwrap_or_default(),
                path,
                fd,
            });
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "no /dev/input/event* advertised BTN_TOOL_PEN + ABS_PRESSURE",
    ))
}

/// Block until at least one event arrives, filling `buf`; returns the count read.
/// A short `read` on an evdev fd always returns whole events, so `n % size == 0`.
pub fn read_events(fd: RawFd, buf: &mut [InputEvent]) -> io::Result<usize> {
    let cap = std::mem::size_of_val(buf);
    // SAFETY: buf is [InputEvent], naturally aligned; we read at most `cap` bytes
    // into it and only interpret the whole events the kernel returned.
    let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, cap) };
    if n < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(n as usize / std::mem::size_of::<InputEvent>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ioctl_request_encoding_matches_c_macros() {
        // Values cross-checked against <linux/input.h> on a 32-bit kernel.
        // EVIOCGNAME(256) = 0x81004506
        assert_eq!(eviocgname(256), 0x8100_4506);
        // EVIOCGBIT(EV_KEY=1, 96) => nr = 0x20+1 = 0x21, size=96=0x60 => 0x80604521
        assert_eq!(eviocgbit(EV_KEY as u32, 96), 0x8060_4521);
        // EVIOCGABS(ABS_X=0) => nr=0x40, size=24=0x18 => 0x80184540
        assert_eq!(eviocgabs(ABS_X as u32), 0x8018_4540);
    }

    #[test]
    fn input_event_is_16_bytes_on_target() {
        // On the 32-bit armv7 device this must be 16; on a 64-bit host it is 24
        // (time_t is 64-bit) — which is exactly why we cross-compile and why the
        // read loop trusts the kernel's own record size, not a host assumption.
        let sz = std::mem::size_of::<InputEvent>();
        assert!(sz == 16 || sz == 24, "unexpected input_event size {sz}");
    }

    #[test]
    fn test_bit_reads_little_endian_bitmask() {
        let bits = [0b0000_0010u8, 0b0000_0001u8]; // bit 1 and bit 8 set
        assert!(!test_bit(&bits, 0));
        assert!(test_bit(&bits, 1));
        assert!(test_bit(&bits, 8));
        assert!(!test_bit(&bits, 9));
        assert!(!test_bit(&bits, 999)); // out of range = false, no panic
    }
}
