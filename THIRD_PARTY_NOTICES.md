# Third-party notices

## Reference implementation

This project is an original, modernized implementation inspired by Section 5,
`section_5_notebooks/machine summarisation-0.ipynb`, from Packt Publishing's
**Advanced NLP Projects with TensorFlow 2.0** repository:

https://github.com/PacktPublishing/Advanced-NLP-Projects-with-TensorFlow-2.0

The reference repository is distributed under the MIT License:

> Copyright (c) 2018 Packt
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

No reference dataset or trained weights are copied from that repository.

## Training dataset

The current model uses the **Amazon Reviews Polarity** title/body corpus assembled
by Xiang Zhang, Junbo Zhao, and Yann LeCun from the Amazon review data introduced
by Julian McAuley and Jure Leskovec. The exact Kaggle distribution used here is
published as **CC0 1.0** and pinned by the downloader to dataset version 2.

- Dataset distribution: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
- Dataset version: `kritanjalijain/amazon-reviews/versions/2`
- Distribution license shown by Kaggle: CC0 1.0
- Dataset paper: https://arxiv.org/abs/1509.01626
- Original review-data paper: https://doi.org/10.1145/2488388.2488466
- CC0 1.0 text: https://creativecommons.org/publicdomain/zero/1.0/

The 1.29 GiB source download and sampled training rows are not redistributed in
this repository. `scripts/download_data.py` reads the 3.6 million-row training
file in chunks and creates the deterministic, deduplicated-before-split sample
described in the model manifest. Dataset users should review the linked source
page and its terms for their own intended use.

## Bootstrap interface styles

The Streamlit interface vendors the CSS-only Bootstrap 5.3.8 distribution in
`assets/bootstrap-5.3.8.min.css`. No Bootstrap JavaScript is loaded.

- Project: https://getbootstrap.com/
- Source: https://github.com/twbs/bootstrap
- Copyright © 2011–2025 The Bootstrap Authors
- License: MIT

The vendored file retains Bootstrap's copyright and license banner. The MIT
permission and warranty terms reproduced in the reference-implementation
notice above also apply to this Bootstrap distribution.
