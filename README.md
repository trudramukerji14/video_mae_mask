# video_mae_mask
In this repositiory, we provide a quick implementation of the video masking strategy from [VideoMAE](https://proceedings.neurips.cc/paper_files/paper/2022/hash/416f9cb3276121c42eebb86352a4354a-Abstract-Conference.html) from scratch.

The masking criterion for VideoMAE implemented "tube-masking" on videos which effectively dealt with issues of temporal correlation and temporal correlation present in video clips leading to a non-trivial pixel reconsruction task. 


<div align="center">
    <img src="images/unmasked.png" alt="unmasked_frames" width="45%">
    <img src="images/masked.png" alt="masked_frames" width="45%">
</div>

You can explore the project's data loading and model implementation interactively in this Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15IRcjmMEvq4mXE00jwRFy8xkJtzX6tUC?usp=sharing)




