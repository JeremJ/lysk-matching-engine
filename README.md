# Matching pipeline code sample (<1k LOC)

This folder mirrors the original file structure so the orchestrator, filters, and compatibility scoring stay readable in context. The snippets come directly from production (minus peripheral CRUD/glue) and demonstrate how we take questionnaire answers, filter candidates, and score compatibility while keeping decision metadata transparent.

## Project context
I built two tracks around this engine: offline events (curated mingling nights) and an experiment with fully online introductions where the system proposed 1:1 dates. The code here is the automated heart of both: it turns questionnaires into profiles, filters for mutual preferences, and then generates ranked suggestions. In practice, it powered sending invite-style suggestions to women about men who fit their stated intent and compatibility profile. Before this, I was mostly building B2B things; this engine was my first revenue-generating B2C product, the scrappy version that proved the idea worked for real users and paid for itself.

## Why I care about this code
I like to think I've written cleaner, more polished code since this project - but this pipeline is special. It is one of the first things I built that earned money online and, more importantly, actually matched people in the real world. The algorithm here helped set up first dates, and a few couples even moved in together after being paired by it. Writing something that tangibly improved lives (and paid my rent) still feels pretty great.

## Included files (kept under 1k LOC)
- `matching_engine/matching_engine.py`: orchestrates questionnaire conversion, filter execution, scoring, and result shaping.
- `matching_engine/filters/filter_factory.py`: assembles the configurable filter pipeline.
- `matching_engine/filters/rules/trait_matching_filter.py`: universal/symmetric trait-matching rule with normalization helpers.
- `matching_engine/scoring/compatibility_scorer.py`: blends embeddings-based life-goals similarity and appearance scores with transparent decision metadata.

External types (e.g., `MatchingProfile`, `MatchResult`, questionnaire adapters, embedding clients) are referenced but omitted to keep the showcase concise. Lines across the four files total ~735 (<1k LOC) so the flow stays readable while showing the “meat” of the system.
