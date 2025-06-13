use std::fmt::{Debug, Display, Formatter};
use std::path;

#[derive(Debug)]
pub enum CokerError {
    IoError(std::io::Error),
    InvalidProject(path::PathBuf, String),
    ScriptError(pyo3::PyErr),
    
}

impl Display for CokerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CokerError::IoError(e) => Debug::fmt(&e, f),
            CokerError::InvalidProject(e, reason) => {
                f.write_fmt(format_args!("Invalid project path: {:?}. {}", e, reason))
            }
            CokerError::ScriptError(e) => {Debug::fmt(&e, f)}
        }
    }
}
impl From<std::io::Error> for CokerError {
    fn from(err: std::io::Error) -> CokerError {
        CokerError::IoError(err)
    }
}
impl From<pyo3::PyErr> for CokerError {
    fn from(err: pyo3::PyErr) -> CokerError {
        CokerError::ScriptError(err)
    }
}
