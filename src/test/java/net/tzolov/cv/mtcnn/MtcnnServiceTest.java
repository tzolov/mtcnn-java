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

import com.fasterxml.jackson.databind.ObjectMapper;
import net.tzolov.cv.mtcnn.json.BoundingBox;
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

		BoundingBox[] boundingBoxes = mtcnnService.faceDetection("classpath:/Anthony_Hopkins_0002.jpg");
		String bboxJson = new ObjectMapper().writeValueAsString(boundingBoxes);

		assertThat(bboxJson, equalTo("[{\"box\":[75,67,95,120],\"confidence\":0.9994938373565674," +
				"\"keypoints\":{\"left_eye\":[101,113],\"right_eye\":[147,113],\"nose\":[124,136],\"mouth_left\":[105,160],\"mouth_right\":[146,160]}}]"));
	}

	@Test
	public void testMultiFaces() throws IOException {

		BoundingBox[] boundingBoxes = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
		String bboxJson = new ObjectMapper().writeValueAsString(boundingBoxes);

		assertThat(bboxJson, equalTo("[{\"box\":[102,152,70,86],\"confidence\":0.9999865293502808," +
				"\"keypoints\":{\"left_eye\":[122,189],\"right_eye\":[154,190],\"nose\":[136,203],\"mouth_left\":[122,219],\"mouth_right\":[151,220]}}," +
				"{\"box\":[332,94,57,69],\"confidence\":0.9992565512657166," +
				"\"keypoints\":{\"left_eye\":[346,120],\"right_eye\":[373,121],\"nose\":[357,134],\"mouth_left\":[346,147],\"mouth_right\":[370,148]}}]"));
	}
}
