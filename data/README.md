# Dataset notes

`sample_social_accounts.csv` is a small, 30-row illustrative dataset included
to make the repository workflow runnable without external access.

## Provenance and intended use

The repository does not record a source, collection protocol, consent basis, or
independent ground-truth review for these rows. Treat the values and `is_fake`
labels as demonstration data only. They are not a representative sample of any
social network, and model scores from this file must not be presented as
real-world accuracy or used to make decisions about real accounts.

## Schema

| Column | Meaning |
| --- | --- |
| `account_age_days` | Account age in days |
| `followers` | Follower count |
| `following` | Following count |
| `posts_count` | Number of posts |
| `has_profile_pic` | Profile picture indicator (`0` or `1`) |
| `has_bio` | Biography indicator (`0` or `1`) |
| `follower_following_ratio` | Followers divided by following plus one |
| `is_fake` | Demo target (`1` = fake, `0` = genuine) |

Counts and ratios must be non-negative; indicator and target columns must be
binary. The ratio must agree with `followers / (following + 1)` within
`0.0005` to allow for three-decimal rounding in the bundled CSV. The
pipeline rejects missing, non-numeric, non-finite, and inconsistent values
and uses a stratified train/test split. Replace this file with a licensed,
representative, independently labeled dataset before drawing operational
conclusions. Do not include personal or platform-restricted data without
appropriate authorization and privacy safeguards.
