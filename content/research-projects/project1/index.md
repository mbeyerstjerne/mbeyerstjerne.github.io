---
title: "Applying Gaussian mixture models to exoplanet data" 
date: 2025-03-21
# tags: ["philology","oleic science","history of oil","Mediterranean world"]
# author: ["Detlev Amadeus Unterholzer","Moritz-Maria von Igelfeld"]
description: "This project examines the applications of Gaussian mixture models to exoplanet population data." 
summary: "This project examines the applications of Gaussian mixture models to exoplanet population data." 
cover:
    image: "exoplanet_clusters-1.png"
    alt: "Some Uses For Olive Oil"
    relative: true
# editPost:
#     URL: "https://github.com/pmichaillat/hugo-website"
#     Text: "Journal of Oleic Science"

---

---

##### Project description

<span style="font-size:90%">This project explores Gaussian mixture models, discussing their effectiveness in detecting clusters in synthetic clustered data, and applying them to exoplanet parameter data taken from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). This was completed as the final project for the course [Advanced Methods in Applied Statistics](https://kurser.ku.dk/course/nfyk15002u), run by Dr. Jason Koskinen in Feb-Apr 2025. </span>

---

##### Abstract

<span style="font-size:90%">In machine learning and statistics, clustering methods are used to group together many-dimensional data points based on the similarity of their features. Gaussian mixture models (GMMs) offer an effective, unsupervised approach for identifying structure in multi-dimensional datasets, making them a valuable tool in exoplanet research where the true data distribution is often unknown. We begin by outlining the mathematical framework of GMMs and testing their effectiveness on synthetic data, where they successfully recover known clusters. When applied to exoplanet period-radius data, we observe clear clustering patterns. However, a two-dimensional Kolmogorov-Smirnov test shows that these clusters do not fully align with a Gaussian mixture, indicating a more complex underlying distribution. Despite this, the GMM provides a meaningful partitioning of the data, revealing structures of interest within the exoplanet population.</span>

---

##### Introduction

<span style="font-size:90%">

The Gaussian mixture model (GMM) is a probabilistic clustering  technique, 
used to identify a set of $K$ clusters in an $M$-dimensional parameter 
space that best represent the observed data distribution, assuming that 
each cluster consists of a multivariate Gaussian distribution. 
This technique is categorized as an unsupervised learning method,
as clusters are identified without requiring prior information
about the data points’ classifications.

</span>

![](exoplanet_clusters-1.png)

---

##### Citation
<!-- 
Unterholzer, Detlev A., and  Moritz-Maria von Igelfeld. 2013. "Unusual Uses For Olive Oil." *Journal of Oleic Science* 34 (1): 449–489. http://www.alexandermccallsmith.com/book/unusual-uses-for-olive-oil.

```latex
@article{UI13,
author = {Detlev A. Unterholzer and Moritz-Maria von Igelfeld},
year = {2013},
title ={Unusual Uses For Olive Oil},
journal = {Journal of Oleic Science},
volume = {34},
number = {1},
pages = {449--489},
url = {http://www.alexandermccallsmith.com/book/unusual-uses-for-olive-oil}}
``` -->

--- -->

##### Related material

+ [Presentation slides](presentation1.pdf)
+ [Summary of the paper](https://www.penguinrandomhouse.com/books/110403/unusual-uses-for-olive-oil-by-alexander-mccall-smith/)
