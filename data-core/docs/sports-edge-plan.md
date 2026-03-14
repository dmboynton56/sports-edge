# Plan: Sports-Edge Standalone Web App

## Goal
Transform the existing "Sports Edge API" script into a high-performance, decoupled, public-facing Next.js platform capable of handling real-time odds and user subscriptions.

## Phase 1: Database & Caching Layer
1. **PostgreSQL Setup:** Create the relational schema for `Users`, `Events/Games`, `Predictions & Edges`, and `Historical Odds`. 
2. **Redis Setup:** Provision a Redis instance.
3. **Backend Pipeline:** Refactor your existing Sports Edge API to:
   - Write permanent historical data (game info, model confidence) to PostgreSQL.
   - Push high-velocity, frequently changing data (live odds, current E[V]) directly to Redis.

## Phase 2: Next.js Frontend Foundation
1. **Scaffold the App:** Initialize a Next.js (App Router) project configured with Tailwind CSS.
2. **State Management:** Set up Zustand for lightweight client-side state (e.g., filtering NFL vs NBA, toggling dark mode).
3. **Data Fetching:** Implement Server-Side Rendering (SSR) to fetch the latest data from Redis on page load, ensuring instant load times and strong SEO for public landing pages.

## Phase 3: UI & Data Visualization
1. **Component Library:** Install Tremor (excellent for dashboards) or Recharts.
2. **The Dashboard:** Build the main feed where users see the "Daily Bias Predictions" and percentage edges.
3. **Historical Charts:** Build individual game pages featuring line charts that map how the line and E[V] have moved over the last 48 hours based on public money.

## Phase 4: Monetization & Launch
1. **Authentication:** Implement user sign-ups (via Supabase Auth or NextAuth).
2. **Stripe Integration:** Create subscription tiers (Free vs. Premium).
3. **Gating Content:** Add logic to the Next.js server components that checks the user's subscription status. Blur or hide the highest-value edges for free users, prompting them to upgrade.
