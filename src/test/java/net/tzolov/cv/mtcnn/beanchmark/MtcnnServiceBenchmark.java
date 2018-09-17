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
package net.tzolov.cv.mtcnn.beanchmark;

import java.io.IOException;

import net.tzolov.cv.mtcnn.FaceAnnotation;
import net.tzolov.cv.mtcnn.MtcnnService;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.runner.RunnerException;

import org.springframework.core.io.DefaultResourceLoader;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceBenchmark {

	@State(Scope.Benchmark)
	public static class ExecutionPlan {
		public MtcnnService mtcnnService;
		public INDArray image;

		@Setup(Level.Trial)
		public void setUp() throws IOException {
			Nd4j.ENFORCE_NUMERICAL_STABILITY = false;

			mtcnnService = new MtcnnService(30, 0.709, new double[] { 0.6, 0.7, 0.7 });

			image = new Java2DNativeImageLoader().asMatrix(
					new DefaultResourceLoader().getResource("classpath:/VikiMaxiAdi.jpg").getInputStream())
					.get(point(0), all(), all(), all()).dup();
		}
	}

	@Fork(value = 1, warmups = 1)
	@Benchmark
	@BenchmarkMode(Mode.AverageTime)
	@Threads(value = 1)
	public void faceDetect(ExecutionPlan plan) throws IOException {
		FaceAnnotation[] faceAnnotations = plan.mtcnnService.faceDetection(plan.image);
	}

	public static void main(String[] args) throws IOException, RunnerException {
		org.openjdk.jmh.Main.main(args);
	}
}
