package net.tzolov.cv.mtcnn.json;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * @author Christian Tzolov
 */
public class Keypoints {

	private int[] leftEye;
	private int[] rightEye;
	private int[] nose;
	private int[] mouthLeft;
	private int[] mouthRight;

	@JsonProperty("left_eye")
	public int[] getLeftEye() {
		return leftEye;
	}

	public void setLeftEye(int[] leftEye) {
		this.leftEye = leftEye;
	}

	@JsonProperty("right_eye")
	public int[] getRightEye() {
		return rightEye;
	}

	public void setRightEye(int[] rightEye) {
		this.rightEye = rightEye;
	}

	@JsonProperty("nose")
	public int[] getNose() {
		return nose;
	}

	public void setNose(int[] nose) {
		this.nose = nose;
	}

	@JsonProperty("mouth_left")
	public int[] getMouthLeft() {
		return mouthLeft;
	}

	public void setMouthLeft(int[] mouthLeft) {
		this.mouthLeft = mouthLeft;
	}

	@JsonProperty("mouth_right")
	public int[] getMouthRight() {
		return mouthRight;
	}

	public void setMouthRight(int[] mouthRight) {
		this.mouthRight = mouthRight;
	}
}
