const STEPS = [
  {
    title: "The model prices it",
    body:
      "Every game and player market gets a probability from a versioned model, published before the event starts. The model version is attached to the pick, so you always know which one made the call.",
  },
  {
    title: "The sportsbook prices it",
    body:
      "We pull the live number from the book and compare. The gap between the two is the edge — and when no book is quoting a market, the row is labelled model-only rather than dressed up as a bet.",
  },
  {
    title: "The result grades it",
    body:
      "The next day the pick is settled against what actually happened and rolled into the season record. Nothing is removed after the fact, which is why some of the numbers on this page are negative.",
  },
];

/**
 * Deliberately unboxed: plain type on the page ground so it reads as prose and
 * leaves the board panel as the only lifted surface. Numbered because these are
 * genuinely sequential — probability, then price, then grade.
 */
export function HowItWorks() {
  return (
    <ol className="grid gap-x-8 gap-y-7 sm:grid-cols-3">
      {STEPS.map((step, index) => (
        <li key={step.title} className="flex flex-col gap-2">
          <span className="figure text-[13px] tracking-[0.08em] text-accent">
            {String(index + 1).padStart(2, "0")}
          </span>
          <h3 className="text-[15px] font-semibold">{step.title}</h3>
          <p className="text-sm leading-relaxed text-muted-foreground">{step.body}</p>
        </li>
      ))}
    </ol>
  );
}
