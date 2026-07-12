//! A synthetic reMarkable Marker, built with **uinput** — the mirror image of `evdev`.
//!
//! ## Why this exists
//! `capture.rs` now refuses to record ink while the **eraser end** of the Marker is in
//! range (`BTN_TOOL_RUBBER`). That path is impossible to exercise on a tablet whose pen is
//! a *basic* Marker: it has no eraser, so the digitizer never emits the event, and the
//! branch is dead — shipped, and untested. Testing it by *not* seeing ink appear proves
//! nothing either: an absence is exactly what a pen with no eraser produces.
//!
//! So: synthesize a pen that *does* have one. `/dev/uinput` lets us register a virtual
//! input device with whatever capabilities we claim, and then feed the kernel events that
//! come back out of `/dev/input/eventN` indistinguishable from real hardware. That means
//! this tests the **real** path — kernel → evdev read → `Capture` — not a mock of it.
//!
//! ## Systems concept: uinput is evdev run backwards
//! Reading a device: `open`, `ioctl(EVIOCGBIT)` to ask what it can do, `read()` structs.
//! *Creating* one: `open("/dev/uinput")`, `ioctl(UI_SET_EVBIT/KEYBIT/ABSBIT)` to declare
//! what it can do, `ioctl(UI_DEV_CREATE)`, then `write()` those same structs. The kernel
//! doesn't care that no silicon is behind it.
//!
//!     fake-pen                 # writes an ERASE stroke, then a PEN stroke
//!
//! Not shipped: the Toltec recipe installs only `ink2tex-rm`.

use std::io;
use std::os::unix::io::AsRawFd;
use std::thread::sleep;
use std::time::Duration;

// --- uinput ABI ------------------------------------------------------------
const UI_SET_EVBIT: libc::c_ulong = 0x4004_5564;
const UI_SET_KEYBIT: libc::c_ulong = 0x4004_5565;
const UI_SET_ABSBIT: libc::c_ulong = 0x4004_5567;
const UI_DEV_CREATE: libc::c_ulong = 0x5501;
const UI_DEV_DESTROY: libc::c_ulong = 0x5502;

const EV_SYN: u16 = 0x00;
const EV_KEY: u16 = 0x01;
const EV_ABS: u16 = 0x03;
const SYN_REPORT: u16 = 0;

const BTN_TOOL_PEN: u16 = 0x140;
const BTN_TOOL_RUBBER: u16 = 0x141;
const BTN_TOUCH: u16 = 0x14a;

const ABS_X: u16 = 0x00;
const ABS_Y: u16 = 0x01;
const ABS_PRESSURE: u16 = 0x18;

/// `struct uinput_user_dev` — name + id + per-axis ranges, written to the uinput fd.
#[repr(C)]
struct UinputUserDev {
    name: [libc::c_char; 80],
    id: libc::input_id,
    ff_effects_max: u32,
    absmax: [i32; 64],
    absmin: [i32; 64],
    absfuzz: [i32; 64],
    absflat: [i32; 64],
}

fn main() -> io::Result<()> {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open("/dev/uinput")?;
    let fd = f.as_raw_fd();

    // Declare what this "device" can do — exactly the rM2 Wacom's advertised set, INCLUDING
    // the eraser end that the physical Marker in the room does not have.
    unsafe {
        libc::ioctl(fd, UI_SET_EVBIT, EV_KEY as libc::c_ulong);
        libc::ioctl(fd, UI_SET_EVBIT, EV_ABS as libc::c_ulong);
        libc::ioctl(fd, UI_SET_EVBIT, EV_SYN as libc::c_ulong);
        for k in [BTN_TOOL_PEN, BTN_TOOL_RUBBER, BTN_TOUCH] {
            libc::ioctl(fd, UI_SET_KEYBIT, k as libc::c_ulong);
        }
        for a in [ABS_X, ABS_Y, ABS_PRESSURE] {
            libc::ioctl(fd, UI_SET_ABSBIT, a as libc::c_ulong);
        }

        let mut dev: UinputUserDev = std::mem::zeroed();
        for (i, b) in b"ink2tex fake pen".iter().enumerate() {
            dev.name[i] = *b as libc::c_char;
        }
        dev.id = libc::input_id {
            bustype: 0x18, // BUS_I2C, same as the real Wacom
            vendor: 0x2d1f,
            product: 0x0095,
            version: 1,
        };
        // Same ranges the real digitizer reports, so the transform behaves identically.
        dev.absmax[ABS_X as usize] = 20966;
        dev.absmax[ABS_Y as usize] = 15725;
        dev.absmax[ABS_PRESSURE as usize] = 4095;

        let bytes = std::slice::from_raw_parts(
            (&dev as *const UinputUserDev) as *const u8,
            std::mem::size_of::<UinputUserDev>(),
        );
        if libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len()) < 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::ioctl(fd, UI_DEV_CREATE) < 0 {
            return Err(io::Error::last_os_error());
        }
    }

    eprintln!("fake-pen: virtual digitizer created (with an eraser end)");
    eprintln!("fake-pen: waiting 3s for the reader to attach…");
    sleep(Duration::from_secs(3));

    // 1. An ERASE stroke: the eraser end is what's in range. Note it still emits BTN_TOUCH
    //    and a full coordinate stream — that is the whole point, and why watching only
    //    BTN_TOUCH records erasing as ink.
    eprintln!("fake-pen: injecting an ERASE stroke (BTN_TOOL_RUBBER + BTN_TOUCH + 40 points)");
    emit(fd, EV_KEY, BTN_TOOL_RUBBER, 1);
    emit(fd, EV_KEY, BTN_TOUCH, 1);
    for i in 0..40 {
        emit(fd, EV_ABS, ABS_X, 4000 + i * 300); // a broad horizontal scrub
        emit(fd, EV_ABS, ABS_Y, 8000);
        emit(fd, EV_ABS, ABS_PRESSURE, 2500);
        emit(fd, EV_SYN, SYN_REPORT, 0);
        sleep(Duration::from_millis(5));
    }
    emit(fd, EV_KEY, BTN_TOUCH, 0);
    emit(fd, EV_KEY, BTN_TOOL_RUBBER, 0);
    emit(fd, EV_SYN, SYN_REPORT, 0);
    sleep(Duration::from_millis(300));

    // 2. A PEN stroke, from the tip, AFTER the flip back. If the gate is over-eager and
    //    latches "eraser" permanently, this stroke disappears too — which would be a worse
    //    bug than the one being fixed.
    eprintln!("fake-pen: injecting a PEN stroke (BTN_TOOL_PEN + BTN_TOUCH + 20 points)");
    emit(fd, EV_KEY, BTN_TOOL_PEN, 1);
    emit(fd, EV_KEY, BTN_TOUCH, 1);
    for i in 0..20 {
        emit(fd, EV_ABS, ABS_X, 10000);
        emit(fd, EV_ABS, ABS_Y, 4000 + i * 200); // a vertical line
        emit(fd, EV_ABS, ABS_PRESSURE, 2500);
        emit(fd, EV_SYN, SYN_REPORT, 0);
        sleep(Duration::from_millis(5));
    }
    emit(fd, EV_KEY, BTN_TOUCH, 0);
    emit(fd, EV_KEY, BTN_TOOL_PEN, 0);
    emit(fd, EV_SYN, SYN_REPORT, 0);

    eprintln!("fake-pen: done — expect exactly 1 stroke (the pen one), 0 from the eraser");
    // Outlive the reader: tearing the device down while it is still in read() gives it
    // ENODEV and it loses the ink it captured.
    sleep(Duration::from_secs(12));
    unsafe { libc::ioctl(fd, UI_DEV_DESTROY) };
    Ok(())
}

fn emit(fd: libc::c_int, kind: u16, code: u16, value: i32) {
    let ev = libc::input_event {
        time: libc::timeval {
            tv_sec: 0,
            tv_usec: 0,
        },
        type_: kind,
        code,
        value,
    };
    unsafe {
        let bytes = std::slice::from_raw_parts(
            (&ev as *const libc::input_event) as *const u8,
            std::mem::size_of::<libc::input_event>(),
        );
        libc::write(fd, bytes.as_ptr() as *const libc::c_void, bytes.len());
    }
}
