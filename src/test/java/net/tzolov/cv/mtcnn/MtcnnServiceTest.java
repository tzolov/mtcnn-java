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
package net.tzolov.cv.mtcnn;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;

import java.io.IOException;
import java.io.InputStream;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;
import org.springframework.util.StreamUtils;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceTest {


	private MtcnnService mtcnnService;

	@Before
	public void before() {
		mtcnnService = new MtcnnService(20, 0.709, new double[] { 0.6, 0.7, 0.7 });
	}

	@Test
	public void testSingeFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/Anthony_Hopkins_0002.jpg");
		assertThat(toJson(faceAnnotations), equalTo("[{\"bbox\":{\"x\":72,\"y\":64,\"w\":101,\"h\":124}," 
			+ "\"confidence\":0.9997498393058777," 
			+ "\"landmarks\":" 
			+ "[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":102,\"y\":113}}," 
			+ "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":149,\"y\":113}}," 
			+ "{\"type\":\"NOSE\",\"position\":{\"x\":125,\"y\":136}}," 
			+ "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":104,\"y\":159}}," 
			+ "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":146,\"y\":160}}]}]"));
	}
	
	@Test
	public void testFailToDetectFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/broken.png");
		assertThat(toJson(faceAnnotations), equalTo("[]"));
	}

	@Test
	public void testMultiFaces() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
		assertThat(faceAnnotations.length, equalTo(3));
		assertThat(toJson(faceAnnotations), equalTo(
			"[{\"bbox\":{\"x\":102,\"y\":155,\"w\":69,\"h\":81},\"confidence\":0.9999865293502808," 
				+ "\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":121,\"y\":188}}," 
				+ "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}}," 
				+ "{\"type\":\"NOSE\",\"position\":{\"x\":135,\"y\":204}},{\"type\":\"MOUTH_LEFT\",\"position\"" 
				+ ":{\"x\":120,\"y\":218}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":148,\"y\":221}}]}," 
				+ "{\"bbox\":{\"x\":333,\"y\":97,\"w\":54,\"h\":65},\"confidence\":0.9999747276306152,\"landmarks\":" 
				+ "[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":120}},{\"type\":\"RIGHT_EYE\",\"position\"" 
				+ ":{\"x\":372,\"y\":120}},{\"type\":\"NOSE\",\"position\":{\"x\":357,\"y\":133}}," 
				+ "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":347,\"y\":147}},{\"type\":\"MOUTH_RIGHT\"," 
				+ "\"position\":{\"x\":369,\"y\":148}}]}]"));
	}


	@Test
	public void testFacesAlignment() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/pivotal-ipo-nyse.jpg");
		assertThat(faceAnnotations.length, equalTo(7));
	}

	@Test
	public void testFaceDetection() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/multiple-faces-5.jpg");
		assertThat(faceAnnotations.length, equalTo(5));
	}
	
	@Test
	public void testFacesAlignment2() throws IOException {
		try (InputStream is = new ClassPathResource("classpath:/MarkPollack.jpg").getInputStream()) {
			byte[] image = StreamUtils.copyToByteArray(is);
			FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(image);
			mtcnnService.faceAlignment(null, faceAnnotations, 44, 160, true);
		}

		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/pivotal-ipo-nyse.jpg");
		assertThat(faceAnnotations.length, equalTo(7));
//		assertThat(toJson(faceAnnotations), equalTo(""));
	}

	private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
		return new ObjectMapper().writeValueAsString(faceAnnotations);
	}
}
