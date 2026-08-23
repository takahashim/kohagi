//! The running program's name.
//!
//! Kohagi prefixes stderr lines with the binary name. Keeping it here makes the
//! prefix consistent across modules.
//!
//! The default is `kohagi` for library users that do not set a name.

use std::sync::OnceLock;

static NAME: OnceLock<&'static str> = OnceLock::new();

/// Sets this program's name before it emits warnings.
///
/// Later calls are ignored because a run has one name.
pub fn set(name: &'static str) {
    let _ = NAME.set(name);
}

/// Returns the prefix for stderr messages.
pub fn name() -> &'static str {
    NAME.get().copied().unwrap_or("kohagi")
}

/// Writes one prefixed line to stderr.
///
/// A macro keeps the call site focused on the message.
macro_rules! remark {
    ($($arg:tt)*) => {
        eprintln!("{}: {}", $crate::program::name(), format_args!($($arg)*))
    };
}

pub(crate) use remark;
