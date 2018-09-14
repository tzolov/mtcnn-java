package net.tzolov.cv.mtcnn.json;

/**
 * @author Christian Tzolov
 */
public class BoundingBox {

	/**
	 * [x, y, w, h]
	 */
	private int[] box;

	/**
	 * computed confidence that there is a face in the box
	 */
	private double confidence;

	/**
	 * Face features coordinates.
	 */
	private Keypoints keypoints;

	public int[] getBox() {
		return box;
	}

	public void setBox(int[] box) {
		this.box = box;
	}

	public double getConfidence() {
		return confidence;
	}

	public void setConfidence(double confidence) {
		this.confidence = confidence;
	}

	public Keypoints getKeypoints() {
		return keypoints;
	}

	public void setKeypoints(Keypoints keypoints) {
		this.keypoints = keypoints;
	}
}
