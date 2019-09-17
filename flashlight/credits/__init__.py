#!/usr/bin/env python3

from duecredit import due, BibTeX

# ugly hack for inserting otherwise missing citations.
# probably not the best way, but i did not find better

due.dcite(
    BibTeX(
        """
@inproceedings{paszke2017automatic,
  title={Automatic Differentiation in {PyTorch}},
  author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
  booktitle={NIPS Autodiff Workshop},
  year={2017}
}"""
    ),
    description='Machine learning - frameworks',
    path='torch',
)(lambda x: x)(0)

due.dcite(
    BibTeX(
        """
@inproceedings{paszke2017automatic,
  title={Automatic Differentiation in {PyTorch}},
  author={Paszke, Adam and Gross, Sam and Chintala, Soumith and Chanan, Gregory and Yang, Edward and DeVito, Zachary and Lin, Zeming and Desmaison, Alban and Antiga, Luca and Lerer, Adam},
  booktitle={NIPS Autodiff Workshop},
  year={2017}
}"""
    ),
    description='Machine learning - frameworks',
    path='torch',
)(lambda x: x)(0)
