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

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * @author Christian Tzolov
 */
@JsonPropertyOrder({"bbox", "confidence", "landmarks" })
public class FaceAnnotation {

	@JsonProperty("bbox")
	private BoundingBox boundingBox;

	/**
	 * computed confidence that there is a face in the box
	 */
	private double confidence;

	/**
	 * Face features coordinates.
	 */
	private Landmark[] landmarks;

	public double getConfidence() {
		return confidence;
	}

	public void setConfidence(double confidence) {
		this.confidence = confidence;
	}

	public BoundingBox getBoundingBox() {
		return boundingBox;
	}

	public void setBoundingBox(BoundingBox boundingBox) {
		this.boundingBox = boundingBox;
	}

	public Landmark[] getLandmarks() {
		return landmarks;
	}

	public void setLandmarks(Landmark[] landmarks) {
		this.landmarks = landmarks;
	}

	public static class Landmark {
		public enum LandmarkType {
			LEFT_EYE,
			RIGHT_EYE,
			NOSE,
			MOUTH_LEFT,
			MOUTH_RIGHT
		}

		public static class Position {
			private int x;
			private int y;

			public int getX() {
				return x;
			}

			public void setX(int x) {
				this.x = x;
			}

			public int getY() {
				return y;
			}

			public void setY(int y) {
				this.y = y;
			}

			public static Position of(int x, int y) {
				Position p = new Position();
				p.setX(x);
				p.setY(y);
				return p;
			}
		}

		private LandmarkType type;

		private Position position;

		public LandmarkType getType() {
			return type;
		}

		public void setType(LandmarkType type) {
			this.type = type;
		}

		public Position getPosition() {
			return position;
		}

		public void setPosition(Position position) {
			this.position = position;
		}

		public static Landmark of(LandmarkType type, Position position) {
			Landmark l = new Landmark();
			l.setPosition(position);
			l.setType(type);
			return l;
		}
	}

	public static class BoundingBox {
		private int x;
		private int y;
		private int w;
		private int h;

		public int getX() {
			return x;
		}

		public void setX(int x) {
			this.x = x;
		}

		public int getY() {
			return y;
		}

		public void setY(int y) {
			this.y = y;
		}

		public int getW() {
			return w;
		}

		public void setW(int w) {
			this.w = w;
		}

		public int getH() {
			return h;
		}

		public void setH(int h) {
			this.h = h;
		}

		public static BoundingBox of(int x, int y, int w, int h) {
			BoundingBox b = new BoundingBox();
			b.setX(x);
			b.setY(y);
			b.setW(w);
			b.setH(h);
			return b;
		}
	}
}
