#!/bin/sh

dir="$(git rev-parse --show-toplevel)"

# Check that there's some commits in the first place
if [[ -n $(git rev-list -n 1 --all) ]]; then
    # Ensure that we're only actually looking at the state of the repo as
    # it's being commited.
    # 
    # --keep-index means staged chunks won't be stashed; -u means untracked
    # files will be stashed.
    if [[ -n $(git status --porcelain | grep -E '^( M|\?\?)') ]]; then
        stashed="yes"
        git stash --keep-index -u
    fi
fi

# Helper function to un-stash any changes, if we had to stash them
unstash () {
    if [[ "$stashed" == "yes" ]]; then
        git stash pop
    fi
}

RUSTDOC="cargo +nightly rustdoc --" # nightly is required for -Z flags

if rg '@''commit-fail' $dir -g "!$0"; then # allow <search term> to mark things that shouldn't be committed
    unstash
    exit 1
elif rg '(debug_println|enable_debug)!' $dir; then
    unstash
    exit 1
elif rg 'auto_fuzz_\d+_unnamed' $dir; then # prevent unnamed fuzz tests
    unstash
    exit 1
elif ! cargo fmt --check; then
    unstash
    exit 1
elif ! $RUSTDOC --document-private-items -Z unstable-options --check; then
    unstash
    exit 1
fi

unstash
