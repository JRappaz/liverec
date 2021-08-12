# LiveRec

This repository contains the code of LiveRec, from the paper  
**Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption**  
by Jérémie Rappaz, Julian McAuley and Karl Aberer, accepted at RecSys 2021

## Abstract
 
Live-streaming platforms broadcast user-generated video in real-time. Recommendation on these platforms shares similarities with traditional settings, such as a large volume of heterogeneous content and highly skewed interaction distributions. However, several challenges must be overcome to adapt recommendation algorithms to live-streaming platforms: first, content availability is dynamic which restricts users to choose from only a subset of items at any given time; during training and inference we must carefully handle this factor in order to properly account for such signals, where 'non-interactions' reflect availability as much as implicit preference. Streamers are also fundamentally different from 'items' in traditional settings: repeat consumption of specific channels plays a significant role, though the content itself is fundamentally ephemeral.

In this work, we study recommendation in this setting of a dynamically evolving set of available items. We propose LiveRec, a self-attentive model that personalizes item ranking based on both historical interactions and current availability. We also show that carefully modelling repeat consumption plays a significant role in model performance. To validate our approach, and to inspire further research on this setting, we release a dataset containing 475M user interactions on Twitch over a 43-day period. We evaluate our approach on a recommendation task and show our method to outperform various strong baselines in ranking the currently available content.

# Datasets

Our datasets have been collected from Twitch. We took a full snapshot of all availble streams every 10 minutes, during 43 days. For each stream, we retrieved all logged in users from the chat. All usernames have been anonymized.


Two datasets are provided in our [Google Drive](https://drive.google.com/drive/folders/1BD8m7a8m7onaifZay05yYjaLxyVV40si?usp=sharing). The file `full_a.csv.gz` contains the full dataset whil `100k.csv` is a subset of 100k users for benchmark purposes.


