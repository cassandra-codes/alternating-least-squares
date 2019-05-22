(ns cost-allocation.data
  (:require [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn open-csv
  [csv-filename]
  (with-open [reader (io/reader csv-filename)]
    (doall
     (csv/read-csv reader))))

(defn save-csv
  [csv-filename data]
  (with-open [writer (io/writer csv-filename)]
    (doall
     (csv/write-csv writer data))))

(defn diff
  "Finds the differences between two vectors"
  [left right]
  (loop [xs (sort left)
         ys (sort right)
         diffs []]
    (cond (and (empty? xs) (empty? ys)) diffs
          (empty? xs) (concat diffs (vec (map vector (repeat :right) ys)))
          (empty? ys) (concat diffs (vec (map vector (repeat :left) xs)))
          :else (let [x (first xs)
                      y (first ys)]
                  (cond (pos? (compare x y)) (recur xs (rest ys) (conj diffs [:right y]))
                        (neg? (compare x y)) (recur (rest xs) ys (conj diffs [:left x]))
                        :else (recur (rest xs) (rest ys) diffs))))))
