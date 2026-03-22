use std::fs;
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;

pub(crate) fn read_to_string(path: &Path) -> io::Result<String> {
    ensure_existing_regular_file(path, "read")?;
    fs::read_to_string(path)
}

pub(crate) fn write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    ensure_regular_file_or_missing(path, "write")?;
    fs::write(path, contents)
}

fn ensure_existing_regular_file(path: &Path, operation: &str) -> io::Result<()> {
    let metadata = fs::metadata(path)?;
    ensure_regular_file(path, &metadata.file_type(), operation)
}

fn ensure_regular_file_or_missing(path: &Path, operation: &str) -> io::Result<()> {
    match fs::metadata(path) {
        Ok(metadata) => ensure_regular_file(path, &metadata.file_type(), operation),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

fn ensure_regular_file(path: &Path, file_type: &fs::FileType, operation: &str) -> io::Result<()> {
    if file_type.is_file() {
        return Ok(());
    }

    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "cannot {operation} {}; expected a regular file, found {}",
            path.display(),
            describe_file_type(file_type)
        ),
    ))
}

fn describe_file_type(file_type: &fs::FileType) -> &'static str {
    if file_type.is_dir() {
        return "directory";
    }
    if file_type.is_symlink() {
        return "symlink";
    }

    #[cfg(unix)]
    {
        if file_type.is_fifo() {
            return "fifo";
        }
        if file_type.is_socket() {
            return "socket";
        }
        if file_type.is_block_device() {
            return "block device";
        }
        if file_type.is_char_device() {
            return "character device";
        }
    }

    "special file"
}
