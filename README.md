# jsalt

> WORK IN PROGRESS -- Information is in the process of being added

## Introduction

This is some of my work at the [2022 Annual Frederick Jelinek Memorial Summer Workshop](https://www.clsp.jhu.edu/2022-eighth-frederick-jelinek-memorial-summer-workshop/).

This repository highlights a pitch reconstruction task which investigates cross-lingual transfer for large pre-trained models trained only on speech data English data.

Audio data is pulled from the [FLERUS dataset](https://huggingface.co/datasets/google/fleurs), and pitch labels are created using [pYAAPT](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html).

## Pitch Reconstruction

Pitch reconstruction starts off with an audio file.

This audio file is then fed into a pre-trained model (in this case HuBERT). The hiddens states (the model's internal representation of the audio) are then extracted from the ouptput. In this project, this was done using the [S3PRL toolkit](https://github.com/s3prl/s3prl).

Separately, a pitch tracking algorithm (in this case, [pYAAPT](http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html), others like kaldi can work as well) is used to create "ground truth" pitch information for the audio file (since FLEURS is not labeled for pitch).

A linear regression of the pYAAPT pitch frames against the hidden states is run, after aligning the frames. Mean Squared Error is calculated on the test set outputs after setting anyhting predicted to be 60Hz to 0 and applying a weak convolution. If the hidden states correlate with the pitch information frame by frame, we can say that the model's internal representation does include some pitch information. 

![Pitch Reconstruction Diagram](https://user-images.githubusercontent.com/55806453/182929477-25266ed3-184a-4ca6-b03e-c28a88afa1e3.png)

## Cross-Lingual Transfer in Pitch Reconstruction

Many of pre-trained models are pre-trained only on Enlgish data. This begs the question: how well do these models capture pitch information from other languages? Does learning to represent pitch in one language transfer over to others?

Here, zero-shot multilingual pitch reconstruction is performed using the above method. The results follow:

## Results

TODO

> All results on HuBERT -- fbank baseline results pending

#### Results For All Languages - Best HuBERT Layer

Interestingly, English [Red] does worse than Average [Black].

![All Language Results](https://user-images.githubusercontent.com/55806453/182918352-28c8e83e-c15c-4ee6-927f-1563d1ca2490.png)

#### Results For All Layers Across Languages

It appears generally true that the earlier layers of the model encode more pitch information. The black bar is the average across layers.

![All Layers Results](https://user-images.githubusercontent.com/55806453/182926553-783ba8f7-b57d-43c7-8a40-f45e3f367d62.png)

#### Trends: Language Group

FLEURS provides language group ids. The results do not seem particularly different among the groups.

![Language Group Results](https://user-images.githubusercontent.com/55806453/182927867-7f784477-aaba-4758-89b6-fe33c223a730.png)

#### Trends: Language Type

There does appear to be minor differences among tonal, pitch accent, and non-tonal languages. Tonal langugaes appear to do slightly better on average.

![Language Type Results](https://user-images.githubusercontent.com/55806453/182929034-f938967a-273f-4dda-bcc2-15904c75688a.png)

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

#### Sorani Kurdish Sample Reconstruction

زستان دەکرێت زۆر سارد بێت پلەی گەرمی زۆر بە دەگمەن بۆ بەستان دادەبەزێت بەڵام با و شێداری یەک دەگرن بۆ ئەوەی هەست بکەیت ساردترە لەوەی گەرمیپێو نیشانی دەدات

https://user-images.githubusercontent.com/55806453/182905384-6d452ed0-168c-4278-9416-bff5f70f6f68.mp4

![Sorani Kurdish Sample Reconstruction](https://user-images.githubusercontent.com/55806453/182905695-07a0968d-7a88-4708-8e2e-0d6f9bd0003a.png)

#### Zulu Sample Reconstruction

ubusika bungaba nzima kakhulu amazing okushisa angase adlulele ngale kweqhwa kodwa umoya kanye nomswakama sekuhlangene kwenza kubande ngaphezu kwalokho isikali sokushisa esikushoyo

https://user-images.githubusercontent.com/55806453/182905829-9f193f08-e764-4740-ba7a-92c41ba9282c.mp4

![Zulu Sample Reconstruction](https://user-images.githubusercontent.com/55806453/182929661-a3d6cf55-7b2a-468f-b9c0-c46f34de16a6.png)



