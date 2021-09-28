# LiveRec

This repository contains the code of LiveRec, from the paper  
**Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption**  
by Jérémie Rappaz, Julian McAuley and Karl Aberer, accepted as a full paper at RecSys 2021

## Abstract
 
Live-streaming platforms broadcast user-generated video in real-time. Recommendation on these platforms shares similarities with traditional settings, such as a large volume of heterogeneous content and highly skewed interaction distributions. However, several challenges must be overcome to adapt recommendation algorithms to live-streaming platforms: first, content availability is dynamic which restricts users to choose from only a subset of items at any given time; during training and inference we must carefully handle this factor in order to properly account for such signals, where 'non-interactions' reflect availability as much as implicit preference. Streamers are also fundamentally different from 'items' in traditional settings: repeat consumption of specific channels plays a significant role, though the content itself is fundamentally ephemeral.

In this work, we study recommendation in this setting of a dynamically evolving set of available items. We propose LiveRec, a self-attentive model that personalizes item ranking based on both historical interactions and current availability. We also show that carefully modelling repeat consumption plays a significant role in model performance. To validate our approach, and to inspire further research on this setting, we release a dataset containing 475M user interactions on Twitch over a 43-day period. We evaluate our approach on a recommendation task and show our method to outperform various strong baselines in ranking the currently available content.

## Datasets

Two datasets are provided in our [Google Drive](https://drive.google.com/drive/folders/1BD8m7a8m7onaifZay05yYjaLxyVV40si?usp=sharing). The file `full_a.csv.gz` contains the full dataset whil `100k.csv` is a subset of 100k users for benchmark purposes.

|                    | Twitch 100k | Twitch full |
|--------------------|-------------|-------------|
| #Users             | 100k        | 15.5M       |
| #Items (streamers) | 162.6k      | 465k        |
| #Interactions      | 3M          | 124M        |
| #Timesteps (10min) | 6148        | 6148        |

Our datasets have been collected from Twitch. We took a full snapshot of all availble streams every 10 minutes, during 43 days. For each stream, we retrieved all logged in users from the chat. All usernames have been anonymized. Start and stop times are provided as integers and represent periods of 10 minutes.

#### Fields description

* `user_id`: user identifier (anonymized).
* `stream id`: stream identifier, could be used to retreive a single broadcast segment (not used in our study). 
* `streamer name`: name of the channel.
* `start time`: first crawling round at which the user was seen in the chat.
* `stop time`: last crawling round at which the user was seen in the chat.

## Credits
If you find any of this useful in your own research, please cite

```
@inproceedings{rappaz2021recommendation,
  title={Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption},
  author={Rappaz, J{\'e}r{\'e}mie and McAuley, Julian and Aberer, Karl},
  booktitle={Fifteenth ACM Conference on Recommender Systems},
  pages={390--399},
  year={2021}
}
```

For more information, please contact Jérémie at `rappaz [dot] jeremie [at] gmail [dot] com`.
