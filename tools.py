import os
import git



def get_latest_commit_info(repos):
    """
    Returns information about the most recent commit in the repository,
    including author, date, and the full commit message.
    """
    latest_commit = repos.head.commit
    author = latest_commit.author.name
    date = latest_commit.committed_datetime.strftime("%Y-%m-%d %H:%M:%S")
    message = latest_commit.message.strip()
    return f"Latest Commit:\nAuthor: {author}\nDate: {date}\nMessage: {message}"


def get_repository_size(repopath):
    """
    Calculates and returns the total size of the repository on disk.
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(repopath):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    
    # Convert size to a human-readable format (MB)
    size_in_mb = total_size / (1024 * 1024)
    return f"Total repository size is: {size_in_mb:.2f} MB"


def get_lines_of_code(repopath):
    """
    Counts the total lines of code (LOC) for common source files (.py, .js, .md)
    in the repository, excluding blank lines.
    """
    total_loc = 0
    # Add any other file extensions you care about
    source_file_extensions = {'.py', '.js', '.ts', '.md', '.html', '.css'}
    
    for dirpath, dirnames, filenames in os.walk(repopath):
        # Exclude the .git directory
        if '.git' in dirnames:
            dirnames.remove('.git')
            
        for filename in filenames:
            if any(filename.endswith(ext) for ext in source_file_extensions):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = [line for line in f.readlines() if line.strip()] # Ignore blank lines
                        total_loc += len(lines)
                except Exception as e:
                    # Ignore files that can't be read
                    pass
    return f"Total lines of code in source files: {total_loc}"