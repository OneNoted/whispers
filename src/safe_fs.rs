use std::fs;
use std::io;
use std::path::Path;

#[cfg(unix)]
use std::io::{Read, Write};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

#[cfg(unix)]
use std::os::unix::fs::FileTypeExt;

pub(crate) fn read_to_string(path: &Path) -> io::Result<String> {
    ensure_existing_regular_file(path, "read")?;

    #[cfg(unix)]
    {
        let mut file = open_for_read(path)?;
        ensure_regular_file(path, &file.metadata()?.file_type(), "read")?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }

    #[cfg(not(unix))]
    {
        fs::read_to_string(path)
    }
}

pub(crate) fn write(path: &Path, contents: impl AsRef<[u8]>) -> io::Result<()> {
    ensure_regular_file_or_missing(path, "write")?;

    #[cfg(unix)]
    {
        let mut file = open_for_write(path)?;
        ensure_regular_file(path, &file.metadata()?.file_type(), "write")?;
        file.write_all(contents.as_ref())?;
        Ok(())
    }

    #[cfg(not(unix))]
    {
        fs::write(path, contents)
    }
}

fn ensure_existing_regular_file(path: &Path, operation: &str) -> io::Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure_regular_file(path, &metadata.file_type(), operation)
}

fn ensure_regular_file_or_missing(path: &Path, operation: &str) -> io::Result<()> {
    match fs::symlink_metadata(path) {
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

#[cfg(unix)]
fn open_for_read(path: &Path) -> io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    options.open(path)
}

#[cfg(unix)]
fn open_for_write(path: &Path) -> io::Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    options.open(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::unique_temp_dir;

    #[cfg(unix)]
    use std::os::unix::fs::symlink;

    #[test]
    fn write_creates_regular_file() {
        let dir = unique_temp_dir("safe-fs-write");
        let path = dir.join("status.json");

        write(&path, b"ok").expect("write should create regular file");

        assert_eq!(fs::read_to_string(path).expect("read written file"), "ok");
    }

    #[cfg(unix)]
    #[test]
    fn read_rejects_symlink_even_when_target_is_regular_file() {
        let dir = unique_temp_dir("safe-fs-read-symlink");
        let target = dir.join("target.txt");
        let link = dir.join("link.txt");
        fs::write(&target, "secret").expect("write target");
        symlink(&target, &link).expect("create symlink");

        let err = read_to_string(&link).expect_err("symlink read should be rejected");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn write_rejects_symlink_even_when_target_is_regular_file() {
        let dir = unique_temp_dir("safe-fs-write-symlink");
        let target = dir.join("target.txt");
        let link = dir.join("link.txt");
        fs::write(&target, "seed").expect("write target");
        symlink(&target, &link).expect("create symlink");

        let err = write(&link, b"updated").expect_err("symlink write should be rejected");

        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(fs::read_to_string(target).expect("read target"), "seed");
    }
}
