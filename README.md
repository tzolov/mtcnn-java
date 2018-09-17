# MTCNN Face Detection for Java, using Tensorflow and ND4J

[ ![Download](https://api.bintray.com/packages/big-data/maven/mtcnn-java/images/download.svg) ](https://bintray.com/big-data/maven/mtcnn-java/_latestVersion)

> Note: This is still Work In Progress!

Java and Tensorflow implementation of the [MTCNN Face Detector](https://arxiv.org/abs/1604.02878). It is based on David Sandberg's [FaceNet's MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align) 
python implementation and the original [Zhang, K et al. (2016) ZHANG2016](https://arxiv.org/abs/1604.02878) paper and related [Matlab implementation](https://github.com/kpzhang93/MTCNN_face_detection_alignment).

It reuses the `PNet`, `RNet` and `ONet` Tensorflow models created in [FaceNet's MTCNN](https://github.com/davidsandberg/facenet/tree/master/src/align) project and 
initialized with the original pre-trained [weights](https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv2/model). Find [here](https://github.com/davidsandberg/facenet/pull/866) 
how to freeze those models.

> Note that all necessary Tensorflow models are already pre-bundled with this project! No need to download or freeze those by yourself.

MTCNN involves a lot of linear algebra computation such as multi-dimensional array computation. Therefore the [ND4J](https://deeplearning4j.org/docs/latest/nd4j-overview) library is used for implementing the 
 processing steps that connect the `PNet`, `RNet` and `ONet` Tensorflow networks. Furthermore the []JavaCV](https://github.com/bytedeco/javacv) is leverage for image manipulation and the ND4J-Tensorflow's GraphRunner is used to 
 infer the pre-trained tensorflow models. Later allows to share the ND4J and JavaCV intermediate processing states directly with the tensorflow runner, off-heap without need of additional serialization/deserialization.        



## Quick Start

Use the following dependency to add the `mtcnn` utility to your project 
```xml
<dependency>
  <groupId>net.tzolov.cv</groupId>
  <artifactId>mtcnn</artifactId>
  <version>0.0.3</version>
</dependency>
```
You may also need to add the following maven repository to your pom:
```xml
<repositories>
    <repository>
        <snapshots>
            <enabled>false</enabled>
        </snapshots>
        <id>bintray-big-data-maven</id>
        <name>bintray</name>
        <url>https://dl.bintray.com/big-data/maven</url>
    </repository>
</repositories>
```

## Benchmar
Use this [MtcnnServiceBenchmark](https://github.com/tzolov/mtcnn-java/blob/master/src/test/java/net/tzolov/cv/mtcnn/beanchmark/MtcnnServiceBenchmark.java) to perform some basic benchmarking. You can change the image URI to test
the performance with different images.   
