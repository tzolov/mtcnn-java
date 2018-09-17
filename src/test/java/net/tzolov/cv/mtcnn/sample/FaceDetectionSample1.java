package net.tzolov.cv.mtcnn.sample;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.MtcnnService;
import net.tzolov.cv.mtcnn.MtcnnUtil;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.springframework.core.io.DefaultResourceLoader;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class FaceDetectionSample1 {

	public static void main(String[] args) throws IOException {

		MtcnnService mtcnnService = new MtcnnService(30, 0.709, new double[] { 0.6, 0.7, 0.7 });

		// Image loading and conversion utilities (part of DataVec)
		Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();
		ObjectMapper jsonMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);

		// Supports file:/, http:/ and classpath: URI prefixes.
		String inputImageUri = "classpath:/pivotal-ipo-nyse.jpg";
		try (InputStream imageInputStream = new DefaultResourceLoader().getResource(inputImageUri).getInputStream()) {


			INDArray image = imageLoader.asMatrix(imageInputStream).get(point(0), all(), all(), all()).dup();

			FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(image);

			System.out.println("Face Annotations (JSON): " + jsonMapper.writeValueAsString(faceAnnotations));

			BufferedImage faceAnnotatedImage = MtcnnUtil.drawFaceAnnotations(imageLoader.asBufferedImage(image), faceAnnotations);

			ImageIO.write(faceAnnotatedImage, "png", new File("./AnnotatedImage.png"));
		}
	}
}
