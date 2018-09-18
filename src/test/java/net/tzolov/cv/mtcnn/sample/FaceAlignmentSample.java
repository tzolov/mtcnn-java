/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.tzolov.cv.mtcnn.sample;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.MtcnnService;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.springframework.core.io.DefaultResourceLoader;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class FaceAlignmentSample {

	public static void main(String[] args) throws IOException {

		MtcnnService mtcnnService = new MtcnnService(20, 0.709, new double[] { 0.6, 0.7, 0.7 });

		// Image loading and conversion utilities (part of DataVec)
		Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();

		// Supports file:/, http:/ and classpath: URI prefixes.
		String inputImageUri = "classpath:/pivotal-ipo-nyse.jpg";
		try (InputStream imageInputStream = new DefaultResourceLoader().getResource(inputImageUri).getInputStream()) {
			INDArray originalImage = imageLoader.asMatrix(imageInputStream).get(point(0), all(), all(), all()).dup();

			FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(originalImage);

			int margin = 44; // margin for the crop around the bounding box (height, width) in pixels.
			int alignedImageSize = 160; // image size for (height, width) in pixels.
			for (int i = 0; i < faceAnnotations.length; i++) {
				INDArray alignedFace = mtcnnService.faceAlignment(originalImage, faceAnnotations[i],
						margin, alignedImageSize, true);
				BufferedImage image = new Java2DNativeImageLoader().asBufferedImage(alignedFace);
				ImageIO.write(image, "png", new File("target/aligned_" + i + ".png"));
				System.out.print(".");
			}
		}
	}
}
