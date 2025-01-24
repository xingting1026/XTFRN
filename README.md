# XTFRN

<p>前情提要，這是一個fewshot, fine-grained的classfy問題</p>
<p>fine-grained就是相似物件的分類，如現在這個資料集就是200種鳥類</p>

<p>--------------</p>

<p>目前遇到的問題:</p>
<p>添加入cl loss之後沒有表現得更好</p>
<p>不確定是在supcontrastive learning上實現上出現問題</p>
<p>還是說transformer在對比學習上沒有更多的效果</p>



<p>--------------</p>

<p>架構</p>
<p>Input: [600, 3, 84, 84]  </p>
<p>-> Backbone: [600, 64, 5, 5]  </p>
<p>-> FSRM: [600, 25, 64]  </p>
<p>--------------</p>

[論文連結](https://arxiv.org/abs/2211.17161v2)

![BiFRN架構圖](image.png)

![FSRM架構圖](image-1.png)

![FMRM架構圖](image-2.png)

![mlp](image-3.png)

![cl loss](image-5.png)