# Project Brief — Detection of Fake Accounts on Social Media

## Snapshot

| Field | Detail |
|-------|--------|
| Project type | B.Tech real-time / research project |
| Author | Amaragani Nikhil Sai (22X31A0513) |
| Institution | Sri Indu Institute of Engineering and Technology (JNTUH) |
| Guide | Mrs. J. Pujitha, Assistant Professor |
| Year | 2023–2024 |
| Domain | Machine Learning, Online Social Networks, Trust & Safety |

## Motivation

Online social networks concentrate personal data and attention, which attracts Sybil attacks, spam, scams, and influence operations. Fake profiles damage user trust and create operational risk for platforms and brands.

## Problem definition

Build an automated system that flags fake accounts while balancing false positives (real users blocked) and false negatives (fakes missed).

## Proposed approach (report)

- Move beyond brittle few-feature filters
- Use richer signals (including spam commenting, engagement rate, artificial activity concepts in the documentation)
- Prefer algorithms that remain useful under missing/incomplete inputs (report discusses Gradient Boosting style advantages vs pure Random Forest assumptions)
- Support continuous improvement via feedback

## Technologies discussed in documentation

- Data collection: Scrapy, BeautifulSoup, Selenium, social APIs
- Storage: SQL / NoSQL / data lakes (vision)
- Processing: Spark/Hadoop-style big-data concepts (vision)
- ML: SVM, Random Forest, Neural Networks, Gradient Boosting concepts

## What this repo proves

A **portfolio-grade supervised learning pipeline** with sample data, preprocessing, multi-model training, evaluation, prediction, and audit logging — easy for a recruiter to clone and run.

## Interview talking points

1. Which features separate genuine vs suspicious accounts?
2. How would you reduce false positives in production?
3. How would you retrain as adversary tactics evolve?
4. What privacy constraints apply when analyzing user data?
