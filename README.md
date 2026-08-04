# Sports Edge

Sports Edge is my sports analytics and prediction project. It started with a
simple NBA question: could I turn public sports data into repeatable, testable
predictions?

## The journey

The first version was a set of notebooks and small scripts. I used them to
explore team ratings, game data, and basic models. The project then grew in
stages:

1. NBA experiments became scheduled prediction jobs.
2. BigQuery became the warehouse for raw data, features, predictions, and
   results.
3. Supabase became the serving layer for the portfolio and dashboard.
4. GitHub Actions and Google Cloud Scheduler made refreshes repeatable.
5. The system expanded to NFL, MLB, MLB home-run markets, PGA, college
   basketball research, and World Cup forecasting.
6. A Next.js dashboard turned the outputs into a product that is easier to
   inspect.

The goal is not only to make a prediction. The goal is to show the full path
from raw data to a monitored public result. Each sport has a different level of
maturity. NBA is the most active surface. Some leagues are still research or
candidate work.

## What it is now

Sports Edge runs Python pipelines that collect data and odds, build features,
run models, and write results to BigQuery. Selected results move to Supabase.
The Next.js dashboard and my portfolio read from that serving layer.

### Built with

- Python
- pandas, NumPy, scikit-learn, LightGBM, XGBoost, and PyTorch
- Google BigQuery
- Supabase and Postgres
- Next.js, React, TypeScript, and Tailwind CSS
- GitHub Actions, Google Cloud Scheduler, and Cloud Run

## Next

The next phase is about clearer model versioning, stronger evaluation, and a
better view of performance across leagues. The system is useful as a working
product, but it is still an active research project.

## Contact

If you want to talk about sports data, machine learning, or a collaboration:

- Email: [dmboynton6@gmail.com](mailto:dmboynton6@gmail.com)
- LinkedIn: [Drew Boynton](https://www.linkedin.com/in/drew-boynton-1bba16180/)
- GitHub: [dmboynton56](https://github.com/dmboynton56)
