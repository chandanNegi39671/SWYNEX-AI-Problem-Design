# Data attribution

**Dataset:** BANKING77, from the PolyAI `task-specific-datasets` repository
**Source:** https://github.com/PolyAI-LDN/task-specific-datasets (folder `banking_data/`)
**Creators:** Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić (PolyAI)
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/
(The licence text is the `LICENSE` file of the source repository; a copy is in `LICENSE-CC-BY-4.0.txt`.)

**Changes made:** none to the text or labels of `train.csv` and `test.csv`. In this project the official
training file is split 80/20 (stratified, seed 42) into train and validation at run time; see `src/train_eval.py`.

**Citation (requested by the authors):**
```bibtex
@inproceedings{Casanueva2020,
    author    = {I{\~{n}}igo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulic},
    title     = {Efficient Intent Detection with Dual Sentence Encoders},
    year      = {2020},
    month     = {mar},
    note      = {Data available at https://github.com/PolyAI-LDN/task-specific-datasets},
    url       = {https://arxiv.org/abs/2003.04807},
    booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020}
}
```
