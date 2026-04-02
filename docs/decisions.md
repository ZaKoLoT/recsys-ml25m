# Technical Decision Log

This document centralizes our fundamental technical choices to ensure consistent evaluation throughout the project.

## 1. Recommendation Type and Business Rules
* **Chosen Approach:** Implicit Recommendation.
* **Validation Threshold (rating_threshold):** 4.0.
* **Strict Rule:** To simplify our dataset (which contains ratings from 0.5 to 5.0 for each user), we will transform this rating system into positive interactions. If a user's rating for a movie is $\ge$ 4.0, the interaction is set to 1 (the user liked the movie). Otherwise, the row is ignored and removed from our dataset. We do not exploit negative feedback.

## 2. Evaluation Protocol
* **User-based Split Method:** We sort each user's history by date (`timestamp`).
  * *Rationale:* This simulates "the future" and prevents data leakage. In a real-world scenario, we predict what a user will watch tomorrow based on their past behavior.
* **Split Parameters (V0):** `N_testV0 = 5`. The last 5 interactions constitute the Test set, the rest constitute the Train set. We ensure a minimum of 3 interactions remain in the training set to maintain model stability. *(Note: for subsequent versions, we will introduce a validation set with `N_val = 2`).*
* **Performance Metrics (Top-K):** Evaluation will be performed on short and medium lists with `K=10` and `K=20`. We will use **Recall@K** and **NDCG@K** metrics.
  * *Why K=10:* This represents a short list (Netflix/Spotify style). It verifies if the system is immediately relevant at the top of the list.
  * *Why K=20:* Provides a more comprehensive and stable evaluation on a slightly longer list.
  * *Why these two:* This is the right compromise (less than 10 is too punitive, more than 20 is unnecessary for the user).
* **Baselines (Week 2):** Basic Popularity model and ALS (Alternating Least Squares for implicit feedback) model.

## 3. Expected Data Deliverables (End of Week 1)
At the end of the ingestion sprint, we will generate the following files in Parquet format:
* **Cleaned Base Files:** `interactions.parquet`, `items.parquet`, `tags.parquet`, `links.parquet`, `genome-scores.parquet`, `genome-tags.parquet`.
* **Control Split Files (V0):** `interactions_train_v0.parquet` and `interactions_test_v0.parquet`.

## 4. Assumptions and Limitations
* **Cold-start:** This initial filtering does not natively handle the cold-start problem (new items or users without history).
* **Temporal Conflicts:** If a user has multiple interactions with strictly identical timestamps, the sorting order for the temporal split may be arbitrary.
