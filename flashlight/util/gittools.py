import git
from datetime import datetime, timezone, timedelta


def git_untracked(repo):
    if len(repo.untracked_files):
        return 'Git untracked files:\n' + ('\n'.join(repo.untracked_files))


def git_changes(repo):
    diff_results = []
    for diff in repo.head.commit.diff(None):
        line = f'   {diff.change_type}    {diff.b_path}'
        diff_results.append(line)
    diff_results.sort()
    if diff_results:
        return 'Git changes:\n' + ('\n'.join(diff_results))


def git_synchronize(repo):
    origin = repo.remotes.origin
    origin.push()


def git_summary(working_dir='.'):
    try:
        repo = git.Repo(working_dir, search_parent_directories=True)
        sha = repo.head.commit.hexsha
        msg = repo.head.commit.message.strip()
        tz = timezone(timedelta(seconds=-repo.head.commit.committer_tz_offset))
        date = datetime.fromtimestamp(repo.head.commit.committed_date, tz=tz)
        date = date.strftime('%Y %b %d, %H:%M:%S %Z')
        dirty = repo.is_dirty()
        if dirty:
            changes = git_changes(repo)
        else:
            changes = None
        return sha, msg, date, dirty, changes
    except git.exc.InvalidGitRepositoryError:
        return None, None, None, False, None
