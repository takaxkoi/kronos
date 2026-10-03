#!/usr/bin/env bash
# Commit forecast results back to the repo (retries if another job pushed first).
set -u
git config user.name "kronos-oracle[bot]"
git config user.email "oracle-bot@users.noreply.github.com"
for p in state site/data data app/config.yaml app/custom_models.yaml; do
  [ -e "$p" ] && git add -A -- "$p"
done
if git diff --cached --quiet; then echo "nothing to commit"; exit 0; fi
git commit -q -m "${1:-oracle update} [skip ci]"
BR="${GITHUB_REF_NAME:-master}"
for i in 1 2 3 4 5 6; do
  if git pull -q --rebase -X theirs origin "$BR"; then
    git push -q origin "HEAD:$BR" && { echo "pushed"; exit 0; }
  else
    git rebase --abort 2>/dev/null || true
  fi
  echo "push retry $i"; sleep $((i * 8))
done
echo "could not push results"; exit 1
