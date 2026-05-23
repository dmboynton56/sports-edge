# Measuring daily-refresh “streak” (scheduled runs)

Week 1 exit criteria call for **≥3 consecutive successful scheduled** [daily-refresh.yml](../../.github/workflows/daily-refresh.yml) runs.

Manual `workflow_dispatch` successes (e.g. run #141) prove the SHA but do not advance the streak counter unless you change acceptance rules.

**GitHub CLI** (authenticated):

```bash
gh run list --repo YOUR_ORG/sports-edge --workflow "Daily Sports-Edge Refresh" \
  --json databaseId,conclusion,event,displayTitle,createdAt \
  --jq '.[] | select(.event=="schedule") | {id:.databaseId, conclusion, createdAt}'
```

Count leading **`conclusion == SUCCESS`** rows from the most recent **scheduled** run backward.

Discord notifications from the workflow remain the day-to-day signal for failures.
