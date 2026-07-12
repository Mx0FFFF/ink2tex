//! Live-ink rendering to the E-Ink panel via libremarkable (device-only).
//!
//! ## Systems concept: E-Ink waveforms
//! An E-Ink refresh is neither free nor one-size-fits-all. A *waveform* is the
//! sequence of voltages that drives pixels from one grey state to another;
//! different waveforms trade quality for speed. For inking we use **DU** (direct
//! update): a fast, 2-level (black/white) partial refresh that makes the ink feel
//! *attached to the pen*. A high-quality `GC16` full refresh (slow, de-ghosts) is
//! for clearing the screen, not for live strokes — using it per-stroke would feel
//! like a laggy mess. We draw each new segment, then fire an **async** DU refresh
//! over just that segment's bounding box so the CPU never blocks on the panel.
//!
//! `docs/device.md`: on the rM2, `/dev/fb0` is not the logical display,
//! so this path relies on rm2fb / libremarkable's swtfb client to reach the EPDC.

use libremarkable::framebuffer::cgmath::Point2;
use libremarkable::framebuffer::common::{
    color, display_temp, dither_mode, waveform_mode, DISPLAYHEIGHT, DISPLAYWIDTH, DRAWING_QUANT_BIT,
};
use libremarkable::framebuffer::core::Framebuffer;
use libremarkable::framebuffer::{FramebufferDraw, FramebufferRefresh, PartialRefreshMode};

pub struct Screen {
    fb: Framebuffer,
}

impl Screen {
    pub fn open() -> Self {
        Screen {
            fb: Framebuffer::new(),
        }
    }

    /// Clear to white with a high-quality full refresh (de-ghost before inking).
    pub fn clear(&mut self) {
        self.fb.clear();
        self.fb.full_refresh(
            waveform_mode::WAVEFORM_MODE_GC16,
            display_temp::TEMP_USE_MAX,
            dither_mode::EPDC_FLAG_USE_DITHERING_PASSTHROUGH,
            0,
            true, // wait: guarantee a clean panel before the first stroke
        );
    }

    /// Draw one ink segment (normalized [0,1] endpoints) and fast-refresh just it.
    pub fn ink_segment(&mut self, from: (f32, f32), to: (f32, f32), pressure: f32) {
        let width = 2 + (pressure.clamp(0.0, 1.0) * 4.0) as u32; // 2..6 px by pressure
        let rect = self.fb.draw_line(px(from), px(to), width, color::BLACK);
        self.fb.partial_refresh(
            &rect,
            PartialRefreshMode::Async, // never block the pen on the panel
            waveform_mode::WAVEFORM_MODE_DU,
            display_temp::TEMP_USE_REMARKABLE_DRAW,
            dither_mode::EPDC_FLAG_EXP1,
            DRAWING_QUANT_BIT,
            false,
        );
    }
}

/// Normalized screen coord → framebuffer pixel.
fn px(n: (f32, f32)) -> Point2<i32> {
    Point2 {
        x: (n.0 * DISPLAYWIDTH as f32) as i32,
        y: (n.1 * DISPLAYHEIGHT as f32) as i32,
    }
}
