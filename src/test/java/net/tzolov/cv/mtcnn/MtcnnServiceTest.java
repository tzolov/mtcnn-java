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

import java.io.IOException;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;

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
		assertThat(toJson(faceAnnotations), equalTo("[{\"bbox\":{\"x\":75,\"y\":67,\"w\":95,\"h\":120}," +
				"\"confidence\":0.9994938373565674," +
				"\"landmarks\":[" +
				"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":101,\"y\":113}}," +
				"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":147,\"y\":113}}," +
				"{\"type\":\"NOSE\",\"position\":{\"x\":124,\"y\":136}}," +
				"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":105,\"y\":160}}," +
				"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":146,\"y\":160}}]}]"));
	}

	@Test
	public void testFailToDetectFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/broken.png");
		assertThat(toJson(faceAnnotations), equalTo("[]"));
	}

	@Test
	public void testMultiFaces() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
		assertThat(toJson(faceAnnotations), equalTo("[{\"bbox\":{\"x\":102,\"y\":152,\"w\":70,\"h\":86}," +
				"\"confidence\":0.9999865293502808," +
				"\"landmarks\":[" +
				"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":122,\"y\":189}}," +
				"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":154,\"y\":190}}," +
				"{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}}," +
				"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":122,\"y\":219}}," +
				"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":151,\"y\":220}}]}," +
				"{\"bbox\":{\"x\":332,\"y\":94,\"w\":57,\"h\":69}," +
				"\"confidence\":0.9992565512657166," +
				"\"landmarks\":[" +
				"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":120}}," +
				"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":373,\"y\":121}}," +
				"{\"type\":\"NOSE\",\"position\":{\"x\":357,\"y\":134}}," +
				"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":346,\"y\":147}}," +
				"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":148}}]}]"));
	}

	private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
		return new ObjectMapper().writeValueAsString(faceAnnotations);
	}
}
