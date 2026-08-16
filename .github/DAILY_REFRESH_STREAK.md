# Measuring daily-refresh “streak” (Cloud Scheduler dispatches)

Week 1 exit criteria call for **≥3 consecutive successful** [daily-refresh.yml](../../.github/workflows/daily-refresh.yml) runs from the external Cloud Scheduler job.

Manual `workflow_dispatch` successes (e.g. run #141) prove the SHA but do not advance the streak counter unless you change acceptance rules.

**GitHub CLI** (authenticated):

```bash
gh run list --repo YOUR_ORG/sports-edge --workflow daily-refresh.yml \
  --json databaseId,conclusion,event,displayTitle,createdAt \
  --jq '.[] | select(.event=="workflow_dispatch") | {id:.databaseId, conclusion, createdAt}'
```

Count leading **`conclusion == SUCCESS`** rows from the most recent Cloud Scheduler dispatch backward. Exclude manually started runs when validating the production streak.

Discord notifications from the workflow remain the day-to-day signal for failures.
