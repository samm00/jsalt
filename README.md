# jsalt

> WORK IN PROGRESS -- Information is in the process of being added

## Introduction

This is some of my work at the [2022 Annual Frederick Jelinek Memorial Summer Workshop](https://www.clsp.jhu.edu/2022-eighth-frederick-jelinek-memorial-summer-workshop/).

This repository highlights a pitch reconstruction task which investigates cross-lingual transfer for large pre-trained models trained only on speech data English data.

Audio data is pulled from the [FLERUS dataset](https://huggingface.co/datasets/google/fleurs), and pitch labels are created using [pYAAPT](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html).

## Pitch Reconstruction

Pitch reconstruction starts off with an audio file.

This audio file is then fed into a pre-trained model (ie. HuBERT, Mockingjay, etc). The hiddens states (the model's internal representation of the audio) are then extracted from the ouptput. In this project, this was done using the [S3PRL toolkit](https://github.com/s3prl/s3prl).

Separately, a pitch tracking algorithm (in this case, [pYAAPT](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html), others like kaldi can work as well) is used to create "ground truth" pitch information for the audio file (since FLEURS is not labeled for pitch).

A linear regression of the pYAAPT pitch frames against the hidden states is run, after aligning the frames. Mean Squared Error is calculated on the test set. If the hidden states correlate with the pitch information frame by frame, we can say that the model's internal representation does include some pitch information. 

![Pitch Reconstruction Diagram](img/pitch_recon.png "Pitch Reconstruction")

## Cross-Lingual Transfer in Pitch Reconstruction

Many of pre-trained models are pre-trained only on Enlgish data. This begs the question: how well do these models capture pitch information from other languages? Does learning to represent pitch in one language transfer over to others?

Here, zero-shot multilingual pitch reconstruction is performed using the above method. The results follow:

## Results

TODO

> Preliminary: Overall, there does not appear to be too much of a difference between other languages and English, with tonal languages prforming slightly better. Numbers and figures to come when the last pieces of data are evaluated.

## Code

#### gather_data.py

Gather data from FLEURS

#### make_pitch_data.py

Generate labels using pYAAPT 

Run with `make_pitch_data.py n`, where `n` is the FLEURS language id 

#### get_hidden_states.py

Use S3PRL to get hidden states for all data.

Called by `linear_regress.py` because the amount of data is so large, it is far quicker to keep in memory than to write and save.

#### linear_regress.py

Peform linear regressions for train/test data for all languages

Run with `linear_regress.py n`, where `n` is the FLEURS language id 

## Data Examples

#### English Sample Reconstruction

the winter can be deceptively chilly temperatures rarely go below freezing but the wind and humidity combine to make it feel colder than what the thermometer says

https://user-images.githubusercontent.com/55806453/182903388-4917748b-f5a6-4ec1-82b7-222100969d8b.mp4

![English Sample Reconstruction](https://user-images.githubusercontent.com/55806453/182904240-96cf09f4-d60f-48fe-8205-5801793721fb.png)

#### Mandarin Sample Reconstruction

冬 天 可 能 会 比 看 上 去 显 得 更 冷 温 度 很 少 低 于 零 度 但 当 风 和 湿 度 结 合 在 一 起 会 让 人 感 觉 比 温 度 计 显 示 的 温 度 要 冷

https://user-images.githubusercontent.com/55806453/182904728-1024badc-bcb1-48fb-a59f-13d6fdf7a7d9.mp4

![Mandarin Sample Reconstruction](https://user-images.githubusercontent.com/55806453/182904507-c3189d39-a9dc-4222-b18f-17a87fa583a3.png)





