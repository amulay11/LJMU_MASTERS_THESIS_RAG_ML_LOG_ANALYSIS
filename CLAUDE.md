# Workspace Rules for Claude

## Git: pushing to origin/main from a diverged master

`master` and `origin/main` have diverged intentionally (different histories).
Every time a temp branch is checked out from `origin/main` and then abandoned,
git removes from the working tree any file that `origin/main` tracks but `master`
does not. This silently deletes local files.

**Rule: never use `git checkout -b <temp> origin/main` to stage a push.**

Use a git worktree instead. A worktree operates in a completely separate
directory and leaves the master working tree untouched.

### Correct pattern for pushing to origin/main

```bash
# 1. Create an isolated worktree from origin/main
git worktree add _remote_push origin/main

# 2. Do all staging/committing inside that directory
cd _remote_push
git add <files>
git commit -m "..."
git push origin HEAD:main
cd ..

# 3. Clean up — master working tree was never touched
git worktree remove _remote_push
```

### Why the temp-branch approach breaks things

`git checkout -b temp origin/main` then `git checkout master` causes git to:
- Add files tracked by origin/main that master does not track (on the way out)
- Remove those same files when returning to master

Files lost this way must be manually restored with:
```bash
git checkout origin/main -- <path>
git restore --staged <path>
```

Using a worktree avoids this entirely.

---

## Git: commit authorship

All commits must be authored by **Abhishek Mulay <abhishekmulay@gmail.com>**.
Never add `Co-Authored-By: Claude` or any Anthropic reference to commit messages.
Git hooks enforce this — do not bypass them with `--no-verify`.

## Git: pushing to remote

Always ask the user for confirmation before running `git push`.

## Code style

All Python code must comply with PEP 8.
E221/E241 violations in intentionally aligned blocks are acceptable.
