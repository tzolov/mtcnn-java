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

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

import javax.imageio.ImageIO;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.Tensor;

import org.springframework.util.Assert;
import org.springframework.util.CollectionUtils;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

/**
 * @author Christian Tzolov
 */
public class MtcnnUtil {

	public static final char C_ORDERING = 'c';
	public static final int CHANNEL_COUNT = 3;
	private static final boolean SORT_ASCENDING = true;

	public enum NonMaxSuppressionType {Min, Union}

	public static List<Double> computeScalePyramid(int height, int width, int minFaceSize, double scaleFactor) {

		double m = (double) 12 / minFaceSize;
		int minLayer = (int) (Math.min(height, width) * m);

		List<Double> scales = new ArrayList<>();

		int factorCount = 0;

		while (minLayer >= 12) {
			scales.add(m * Math.pow(scaleFactor, factorCount));
			minLayer = (int) (minLayer * scaleFactor);
			factorCount++;
		}

		return scales;
	}

	public static PadResult pad(INDArray totalBoxes, int w, int h) {

		// compute the padding coordinates (pad the bounding boxes to square)
		//        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
		//        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
		//        numbox = total_boxes.shape[0]
		INDArray tmpW = Transforms.floor(totalBoxes.get(all(), point(2)).sub(totalBoxes.get(all(), point(0))).add(1));
		INDArray tmpH = Transforms.floor(totalBoxes.get(all(), point(3)).sub(totalBoxes.get(all(), point(1))).add(1));
		long numBox = totalBoxes.shape()[0]; // == totalBoxes.size(0);

		// dx = np.ones(numbox, dtype=np.int32)
		// dy = np.ones(numbox, dtype=np.int32)
		// edx = tmpw.copy().astype(np.int32)
		//  edy = tmph.copy().astype(np.int32)
		INDArray dx = Nd4j.ones(numBox);
		INDArray dy = Nd4j.ones(numBox);
		INDArray edx = tmpW;
		INDArray edy = tmpH;

		// x = total_boxes[:, 0].copy().astype(np.int32)
		// y = total_boxes[:, 1].copy().astype(np.int32)
		// ex = total_boxes[:, 2].copy().astype(np.int32)
		// ey = total_boxes[:, 3].copy().astype(np.int32)
		INDArray x = Transforms.floor(totalBoxes.get(all(), point(0)));
		INDArray y = Transforms.floor(totalBoxes.get(all(), point(1)));
		INDArray ex = Transforms.floor(totalBoxes.get(all(), point(2)));
		INDArray ey = Transforms.floor(totalBoxes.get(all(), point(3)));

		// tmp = np.where(ex > w)
		// edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
		// ex[tmp] = w
		INDArray tmp = getIndexWhereVector(ex, value -> value > w);
		//INDArray tmp = getIndexWhereVector2(ex, Conditions.greaterThan(w));

		if (!tmp.isEmpty()) {
			INDArray b = ex.get(tmp).rsub(w).add(tmpW.get(tmp));
			if (b.isScalar()) {
				edx = edx.putScalar(tmp.toLongVector(), b.getInt(0));
				ex = ex.putScalar(tmp.toLongVector(), w);
			}
			else {
				INDArray updateValue = Nd4j.expandDims(b, 1);
				edx = edx.put(toUpdateIndex(tmp), updateValue);
				ex = ex.put(toUpdateIndex(tmp), Nd4j.zerosLike(tmp).add(w));
			}
		}

		// tmp = np.where(ey > h)
		// edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
		// ey[tmp] = h
		tmp = getIndexWhereVector(ey, value -> value > h);
		//tmp = getIndexWhereVector2(ey, Conditions.greaterThan(h));
		if (!tmp.isEmpty()) {
			INDArray b = ey.get(tmp).rsub(h).add(tmpH.get(tmp));
			if (b.isScalar()) {
				edy = edy.putScalar(tmp.toLongVector(), b.getInt(0));
				ey = ey.putScalar(tmp.toLongVector(), h);
			}
			else {
				INDArray updateValues = Nd4j.expandDims(b, 1);
				edy = edy.put(toUpdateIndex(tmp), updateValues);
				ey = ey.put(toUpdateIndex(tmp), Nd4j.zerosLike(tmp).add(h));
				//ey = ey.put(toUpdateIndex(tmp), h); // BUG
			}
		}

		//  tmp = np.where(x < 1)
		//  dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
		//  x[tmp] = 1
		tmp = getIndexWhereVector(x, value -> value < 1);
		//tmp = getIndexWhereVector2(x, Conditions.lessThan(1));
		if (!tmp.isEmpty()) {
			INDArray b = x.get(tmp).rsub(2);
			if (b.isScalar()) {
				dx.putScalar(tmp.toLongVector(), b.getInt(0));
				x = x.putScalar(tmp.toLongVector(), 1);
			}
			else {
				INDArray updateValues = Nd4j.expandDims(x.get(tmp).rsub(2), 1);
				dx.put(toUpdateIndex(tmp), updateValues);
				// x.put(toUpdateIndex(tmp), 1); // BUG
				x = x.put(toUpdateIndex(tmp), Nd4j.onesLike(tmp));
			}
		}

		// tmp = np.where(y < 1)
		// dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
		// y[tmp] = 1
		tmp = getIndexWhereVector(y, value -> value < 1);
		//tmp = getIndexWhereVector2(y, Conditions.lessThan(1));
		if (!tmp.isEmpty()) {
			INDArray b = y.get(tmp).rsub(2);
			if (b.isScalar()) {
				dy.putScalar(tmp.toLongVector(), b.getInt(0));
				//y.put(toUpdateIndex(tmp), 1); // BUG
				y = y.putScalar(tmp.toLongVector(), 1);
			}
			else {
				INDArray updateValues = Nd4j.expandDims(b, 1);
				dy.put(toUpdateIndex(tmp), updateValues);
				y = y.put(toUpdateIndex(tmp), Nd4j.onesLike(tmp));
				//y.put(toUpdateIndex(tmp), 1); // BUG
			}
		}

		return new PadResult(dy, edy, dx, edx, y, ey, x, ex, tmpW, tmpH);
	}

	private static INDArrayIndex[] toUpdateIndex(INDArray array) {
		return new INDArrayIndex[] { new SpecifiedIndex(array.toLongVector()) };
	}

	/**
	 * Calibrate bounding boxes
	 *
	 * original code:
	 *  - https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/code/codes/MTCNNv2/bbreg.m
	 *  - https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py#L646
	 * @param boundingBox
	 * @param reg
	 * @return
	 */
	public static INDArray bbreg(INDArray boundingBox, INDArray reg) {

		// if reg.shape[1] == 1:
		//    reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))
		if (reg.shape()[1] == 1) {
			//reg = reg.reshape(reg.shape()[2], reg.shape()[3]);
			reg = reg.transpose();
		}

		// w = boundingbox[:, 2] - boundingbox[:, 0] + 1
		// h = boundingbox[:, 3] - boundingbox[:, 1] + 1
		// b1 = boundingbox[:, 0] + reg[:, 0] * w
		// b2 = boundingbox[:, 1] + reg[:, 1] * h
		// b3 = boundingbox[:, 2] + reg[:, 2] * w
		// b4 = boundingbox[:, 3] + reg[:, 3] * h
		INDArray w = boundingBox.get(all(), point(2)).sub(boundingBox.get(all(), point(0))).addi(1);
		INDArray h = boundingBox.get(all(), point(3)).sub(boundingBox.get(all(), point(1))).addi(1);
		INDArray b1 = boundingBox.get(all(), point(0)).add(reg.get(all(), point(0)).mul(w)).transpose();
		INDArray b2 = boundingBox.get(all(), point(1)).add(reg.get(all(), point(1)).mul(h)).transpose();
		INDArray b3 = boundingBox.get(all(), point(2)).add(reg.get(all(), point(2)).mul(w)).transpose();
		INDArray b4 = boundingBox.get(all(), point(3)).add(reg.get(all(), point(3)).mul(h)).transpose();

		// boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
		boundingBox.put(new INDArrayIndex[] { all(), interval(0, 4) }, Nd4j.vstack(b1, b2, b3, b4).transpose());
		return boundingBox;
	}

	/**
	 * Convert the bbox into square.
	 *
	 * original code:
	 *  - https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/code/codes/MTCNNv2/rerec.m
	 *  - https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py#L646
	 *
	 * @param bbox
	 * @param withFloor
	 * @return Returns array representing the squared bbox
	 */
	public static INDArray rerec(INDArray bbox, boolean withFloor) {
		// convert bbox to square
		INDArray h = bbox.get(all(), point(3)).sub(bbox.get(all(), point(1)));
		INDArray w = bbox.get(all(), point(2)).sub(bbox.get(all(), point(0)));
		INDArray l = Transforms.max(w, h);

		bbox.put(new INDArrayIndex[] { all(), point(0) }, bbox.get(all(), point(0)).add(w.mul(0.5)).sub(l.mul(0.5)));
		bbox.put(new INDArrayIndex[] { all(), point(1) }, bbox.get(all(), point(1)).add(h.mul(0.5)).sub(l.mul(0.5)));
		INDArray lTile = Nd4j.repeat(l, 2).transpose();
		bbox.put(new INDArrayIndex[] { all(), interval(2, 4) }, bbox.get(all(), interval(0, 2)).add(lTile));

		if (withFloor) {
			bbox.put(new INDArrayIndex[] { all(), interval(0, 4) }, Transforms.floor(bbox.get(all(), interval(0, 4))));
		}

		return bbox;
	}

	/**
	 * Non Maximum Suppression - greedily selects the boxes with high confidence. Keep the boxes that have overlap area
	 * below the threshold and discards the others.
	 *
	 * original code:
	 *  - https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/code/codes/MTCNNv2/nms.m
	 *  - https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py#L687
	 *
	 * @param boxes nd array with bounding boxes: [[x1, y1, x2, y2 score]]
	 * @param threshold NMS threshold -  retain overlap <= thresh
	 * @param nmsType NMS method to apply. Available values ('Min', 'Union')
	 * @return Returns the NMS result
	 */
	public static INDArray nonMaxSuppression(INDArray boxes, double threshold, NonMaxSuppressionType nmsType) {

		if (boxes.isEmpty()) {
			return Nd4j.empty();
		}

		// TODO Try to prevent following duplications!
		INDArray x1 = boxes.get(all(), point(0)).dup();
		INDArray y1 = boxes.get(all(), point(1)).dup();
		INDArray x2 = boxes.get(all(), point(2)).dup();
		INDArray y2 = boxes.get(all(), point(3)).dup();
		INDArray s = boxes.get(all(), point(4)).dup();

		//area = (x2 - x1 + 1) * (y2 - y1 + 1)
		INDArray area = (x2.sub(x1).add(1)).mul(y2.sub(y1).add(1));

		// sorted_s = np.argsort(s)
		INDArray sortedS = Nd4j.sortWithIndices(s, 0, SORT_ASCENDING)[0];

		INDArray pick = Nd4j.zerosLike(s);
		int counter = 0;

		while (sortedS.size(0) > 0) {

			if (sortedS.size(0) == 1) {
				pick.put(counter++, sortedS.dup());
				break;
			}

			long lastIndex = sortedS.size(0) - 1;
			INDArray i = sortedS.get(point(lastIndex), all()); // last element
			INDArray idx = sortedS.get(interval(0, lastIndex), all()).transpose(); // all until last excluding
			pick.put(counter++, i.dup());

			INDArray xx1 = Transforms.max(x1.get(idx), x1.get(i).getInt(0));
			INDArray yy1 = Transforms.max(y1.get(idx), y1.get(i).getInt(0));
			INDArray xx2 = Transforms.min(x2.get(idx), x2.get(i).getInt(0));
			INDArray yy2 = Transforms.min(y2.get(idx), y2.get(i).getInt(0));

			// w = np.maximum(0.0, xx2 - xx1 + 1)
			// h = np.maximum(0.0, yy2 - yy1 + 1)
			// inter = w * h
			INDArray w = Transforms.max(xx2.sub(xx1).add(1), 0.0f);
			INDArray h = Transforms.max(yy2.sub(yy1).add(1), 0.0f);
			INDArray inter = w.mul(h);

			// if method is 'Min':
			//   o = inter / np.minimum(area[i], area[idx])
			// else:
			//   o = inter / (area[i] + area[idx] - inter)
			int areaI = area.get(i).getInt(0);
			INDArray o = (nmsType == NonMaxSuppressionType.Min) ?
					inter.div(Transforms.min(area.get(idx), areaI)) :
					inter.div(area.get(idx).add(areaI).sub(inter));

			INDArray oIdx = MtcnnUtil.getIndexWhereVector(o, value -> value <= threshold);
			//INDArray oIdx = getIndexWhereVector2(o, Conditions.lessThanOrEqual(threshold));

			if (oIdx.isEmpty()) {
				break;
			}

			sortedS = Nd4j.expandDims(sortedS.get(oIdx), 0).transpose();
		}

		//pick = pick[0:counter]
		return (counter == 0) ? Nd4j.empty() : pick.get(interval(0, counter));
	}


	/**
	 * Use heatmap to generate bounding boxes.
	 *
	 * original code:
	 *  - https://github.com/kpzhang93/MTCNN_face_detection_alignment/blob/master/code/codes/MTCNNv2/generateBoundingBox.m
	 *  - https://github.com/davidsandberg/facenet/blob/master/src/align/detect_face.py#L660
	 *
	 * @param imap
	 * @param reg
	 * @param scale
	 * @param stepThreshold
	 * @return Returns the generated bboxes
	 */
	public static INDArray[] generateBoundingBox(INDArray imap, INDArray reg, double scale, double stepThreshold) {

		int stride = 2;
		int cellSize = 12;

		// imap = np.transpose(imap)
		// y, x = np.where(imap >= t)
		// imap = imap.transpose();
		INDArray bb = MtcnnUtil.getIndexWhereMatrix(imap, v -> v >= stepThreshold);
		//INDArray bb = MtcnnUtil.getIndexWhere3(imap, Conditions.greaterThanOrEqual(stepThreshold));

		if (bb.isEmpty()) {
			return new INDArray[] { Nd4j.empty(), Nd4j.empty() };
		}

		INDArray yx = bb.transpose();

		// TODO : implement the following code fragment
		//  if y.shape[0] == 1:
		//    dx1 = np.flipud(dx1)
		//    dy1 = np.flipud(dy1)
		//    dx2 = np.flipud(dx2)
		//    dy2 = np.flipud(dy2)
		if (yx.size(0) == 1) {
			throw new IllegalStateException("TODO");
		}

		//    q1 = np.fix((stride*bb+1)/scale)
		//    q2 = np.fix((stride*bb+cellsize-1+1)/scale)
		INDArray q1 = Transforms.floor(bb.mul(stride).add(1).div(scale));
		INDArray q2 = Transforms.floor(bb.mul(stride).add(cellSize).div(scale));

		//    dx1 = np.transpose(reg[:,:,0])
		//    dy1 = np.transpose(reg[:,:,1])
		//    dx2 = np.transpose(reg[:,:,2])
		//    dy2 = np.transpose(reg[:,:,3])
		INDArray dx1 = reg.get(all(), all(), point(0));
		INDArray dy1 = reg.get(all(), all(), point(1));
		INDArray dx2 = reg.get(all(), all(), point(2));
		INDArray dy2 = reg.get(all(), all(), point(3));

		// reg = np.transpose(np.vstack([ dx1[(y,x)], dy1[(y,x)], dx2[(y,x)], dy2[(y,x)] ]))
		INDArray outReg = Nd4j.vstack(dx1.get(yx), dy1.get(yx), dx2.get(yx), dy2.get(yx)).transpose();

		//  if reg.size == 0:
		//    reg = np.empty(shape=(0, 3))
		if (outReg.isEmpty()) {
			outReg = Nd4j.empty();
		}

		INDArray score = imap.get(yx).transpose();

		INDArray boundingBox = Nd4j.hstack(q1, q2, score, outReg);

		return new INDArray[] { boundingBox, outReg };
	}

	//public static INDArray getIndexWhereVector2(INDArray input, Condition condition) {
	//	try {
	//		return Nd4j.where(input.match(1, condition), null, null)[0];
	//	}
	//	catch (ND4JIllegalStateException nd4jise) {
	//		return Nd4j.empty();
	//	}
	//}

	/**
	 * Manual (ineffient) implementation of where_np (https://github.com/deeplearning4j/deeplearning4j/issues/6184) for vector input
	 * @param input
	 * @param predicate
	 * @return Returns the Where indexes
	 */
	public static INDArray getIndexWhereVector(INDArray input, Predicate<Double> predicate) {

		Assert.isTrue(input.isVector() || input.isScalar(),
				"Only vectors are accepted but found: " + input.rank());

		List<Integer> indexes = new ArrayList<>();
		for (int i = 0; i < input.size(0); i++) {
			if (predicate.test(input.getDouble(i))) {
				indexes.add(i);
			}
		}

		return CollectionUtils.isEmpty(indexes) ? Nd4j.empty(DataBuffer.Type.FLOAT) : Nd4j.create(indexes);
	}

	/**
	 * Manual (ineffient) implementation of where_np (https://github.com/deeplearning4j/deeplearning4j/issues/6184) for matrix input.
	 *
	 * @param input
	 * @param predicate
	 * @return Returns the where matrix indexes
	 */
	public static INDArray getIndexWhereMatrix(INDArray input, Predicate<Double> predicate) {

		Assert.isTrue(input.isMatrix(), "Expected matrix but found: " + input.rank());

		List<Float> yxIndexList = new ArrayList<>();
		for (int y = 0; y < input.rows(); y++) {
			for (int x = 0; x < input.columns(); x++) {
				double v = input.getDouble(y, x);
				if (predicate.test(v)) {
					yxIndexList.add((float) y);
					yxIndexList.add((float) x);
				}
			}
		}

		if (CollectionUtils.isEmpty(yxIndexList)) {
			return Nd4j.empty();
		}

		return Nd4j.create(yxIndexList).reshape(new int[] { yxIndexList.size() / 2, 2 });
	}

	//public static INDArray getIndexWhere3(INDArray input, Condition condition) {
	//	INDArray mask = input.match(1, condition);
	//	if (mask.maxNumber().intValue() == 0) {
	//		return Nd4j.empty();
	//	}
	//	INDArray[] indexes = Nd4j.where(mask, null, null);
	//	return Nd4j.hstack(Nd4j.expandDims(indexes[0], 1), Nd4j.expandDims(indexes[1], 1));
	//}

	public static INDArray append(INDArray arr1, INDArray values, int dimension) {
		if (dimension == -1) {
			return Nd4j.toFlattened(arr1, values);
		}
		else {
			return Nd4j.concat(dimension, arr1, values);
		}
	}

	/**
	 * Converts ND4J array into a {@link Tensor}
	 * @param indArray {@link INDArray} to covert
	 * @return Returns Float {@link Tensor}
	 */
	public static Tensor<Float> toTensor(INDArray indArray) {
		return Tensor.create(indArray.shape(), FloatBuffer.wrap(indArray.data().asFloat()));
	}

	/**
	 * Converts a Tensorflow {@link Tensor} into an ND4J float array
	 * @param tensor input tensor
	 * @return Returns ND4J representation for the input tensor
	 */
	public static INDArray toNDArray(Tensor<?> tensor) {
		FloatBuffer floatBuffer = FloatBuffer.allocate(tensor.numElements());
		tensor.writeTo(floatBuffer);
		return Nd4j.create(floatBuffer.array(), ArrayUtil.toInts(tensor.shape()), C_ORDERING);
	}

	/**
	 * Converts totalBoxes array into {@link FaceAnnotation} and {@link Keypoints} domain json, appropriate for JSON serialization
	 *
	 * @param totalBoxes input matrix with computed bounding boxes. Each row represents a separate bbox.
	 * @param points input matrix with computed key points. Each row represents a set of keypoints for a bbox having the same row.
	 * @return Returns {@link FaceAnnotation} array representing the detected faces and their {@link Keypoints}.
	 */
	public static FaceAnnotation[] toFaceAnnotation(INDArray totalBoxes, INDArray points) {

		if (totalBoxes.isEmpty()) {
			return new FaceAnnotation[0];
		}

		Assert.isTrue(totalBoxes.rows() == points.rows(), "Inconsistent number of boxes and points");

		FaceAnnotation[] faceAnnotations = new FaceAnnotation[totalBoxes.rows()];
		for (int i = 0; i < totalBoxes.rows(); i++) {
			FaceAnnotation faceAnnotation = new FaceAnnotation();

			faceAnnotation.setBoundingBox(FaceAnnotation.BoundingBox.of(totalBoxes.getInt(i, 0), // x
					totalBoxes.getInt(i, 1), // y
					totalBoxes.getInt(i, 2) - totalBoxes.getInt(i, 0), // w
					totalBoxes.getInt(i, 3) - totalBoxes.getInt(i, 1))); //h

			faceAnnotation.setConfidence(totalBoxes.getDouble(i, 4));

			faceAnnotation.setLandmarks(new FaceAnnotation.Landmark[5]);
			faceAnnotation.getLandmarks()[0] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.LEFT_EYE, FaceAnnotation.Landmark.Position.of(points.getInt(i, 0), points.getInt(i, 5)));
			faceAnnotation.getLandmarks()[1] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.RIGHT_EYE, FaceAnnotation.Landmark.Position.of(points.getInt(i, 1), points.getInt(i, 6)));
			faceAnnotation.getLandmarks()[2] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.NOSE, FaceAnnotation.Landmark.Position.of(points.getInt(i, 2), points.getInt(i, 7)));
			faceAnnotation.getLandmarks()[3] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.MOUTH_LEFT, FaceAnnotation.Landmark.Position.of(points.getInt(i, 3), points.getInt(i, 8)));
			faceAnnotation.getLandmarks()[4] = FaceAnnotation.Landmark.of(FaceAnnotation.Landmark.LandmarkType.MOUTH_RIGHT, FaceAnnotation.Landmark.Position.of(points.getInt(i, 4), points.getInt(i, 9)));

			faceAnnotations[i] = faceAnnotation;
		}

		return faceAnnotations;
	}

	/**
	 * Crops a (x1, y1, x2, y2) box from an input image. Input and output images are represented by NDArray [C, H, W]
	 * @param image image to crop using [C,H,W]
	 * @param x1 cropped image top left X
	 * @param x2 cropped image bottom right X
	 * @param y1 cropped image top left Y
	 * @param y2 cropped image bottom right Y
	 * @return Subset of the input image ndarray that represents the cropped region.
	 */
	public static INDArray crop(INDArray image, int x1, int x2, int y1, int y2) {
		// Expects [C, H, W] dimensions
		Assert.isTrue(image.size(0) == CHANNEL_COUNT, "Input image is expected to have the [3, W, H] dimensions");
		INDArray cropImage = image.dup().get(all(), interval(y1, y2), interval(x1, x2));
		//[C, H, W]
		return cropImage;
	}


	// def prewhiten(x):
	//    mean = np.mean(x)
	//    std = np.std(x)
	//    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
	//    y = np.multiply(np.subtract(x, mean), 1/std_adj)
	//    return y

	/**
	 *
	 * @param image format [Batch, Channel, ]
	 * @return returns the result of the pre-whiten filtering
	 */
	public static INDArray preWhiten(INDArray image) {
		INDArray mean = Nd4j.mean(image);
		INDArray std = Nd4j.std(image);
		INDArray stdAdj = Transforms.max(std, 1.0 / Math.sqrt(image.length()));
		return image.sub(mean).mul(stdAdj.rdiv(1));
	}

	/**
	 * The {@link #pad(INDArray, int, int)} response type
	 */
	public static class PadResult {

		private final INDArray dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph;

		public PadResult(INDArray dy, INDArray edy, INDArray dx, INDArray edx, INDArray y, INDArray ey, INDArray x, INDArray ex, INDArray tmpw, INDArray tmph) {
			this.dy = dy;
			this.edy = edy;
			this.dx = dx;
			this.edx = edx;
			this.y = y;
			this.ey = ey;
			this.x = x;
			this.ex = ex;
			this.tmpw = tmpw;
			this.tmph = tmph;
		}

		public INDArray getDy() {
			return dy;
		}

		public INDArray getEdy() {
			return edy;
		}

		public INDArray getDx() {
			return dx;
		}

		public INDArray getEdx() {
			return edx;
		}

		public INDArray getY() {
			return y;
		}

		public INDArray getEy() {
			return ey;
		}

		public INDArray getX() {
			return x;
		}

		public INDArray getEx() {
			return ex;
		}

		public INDArray getTmpw() {
			return tmpw;
		}

		public INDArray getTmph() {
			return tmph;
		}
	}


	/**
	 * Convert {@link BufferedImage} to byte array.
	 *
	 * @param image the image to be converted
	 * @param format the output image format
	 * @return New array of bytes
	 */
	public static byte[] toByteArray(BufferedImage image, String format) {
		ByteArrayOutputStream baos = new ByteArrayOutputStream();

		try {
			ImageIO.write(image, format, baos);
			byte[] bytes = baos.toByteArray();
			return bytes;
		}
		catch (IOException e) {
			throw new IllegalStateException(e);
		}
		finally {
			try {
				baos.close();
			}
			catch (IOException e) {
				throw new IllegalStateException(e);
			}
		}
	}

	/**
	 *
	 * @param bufferedImage
	 * @return flat byte array representing the buffered image
	 */
	public static byte[] toByteArray(BufferedImage bufferedImage) {
		return ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();
	}

	public static float[] imageByteToFloatArray(byte[] imageBytes) {
		float[] fa = new float[imageBytes.length];
		for (int i = 0; i < imageBytes.length; i++) {
			fa[i] = imageBytes[i] & 0xFF;
		}
		return fa;
	}

}
