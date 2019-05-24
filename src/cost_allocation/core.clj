(ns cost-allocation.core
  (:require [cost-allocation.data :refer :all]
            [clojure.math.numeric-tower :refer [round, abs]]))

(defn cmap
  "Apply function `f` to every combination of x and y in `xs` and `ys`,
  returning the result as a matrix"
  [f xs ys]
  (mapv (fn [x] (mapv (fn [y] (f x y)) ys)) xs))

(defn dot
  "The dot product of two vectors"
  [xs ys]
  (->> [xs ys]
       (apply map *)
       (reduce +)))

(defn diff-vector
  "Return a vector that is element-wise difference between `a` and `b`"
  [a b]
  (mapv - a b))

(defn L2-norm
  "The square root of the sums of squares of an vector"
  [xs]
  (Math/sqrt (dot xs xs)))

(defn round-in
  "Round the values of a matrix.
   Assumes a vector of vectors of numeric values."
  [matrix]
  (mapv (fn [row] (mapv #(-> % round int) row)) matrix))

(defn transpose
  "Transpose an n x m matrix to yield an m x n matrix"
  [matrix]
  (apply mapv vector matrix))

(defn sum-rows
  "A vector of the sums of each row of a matrix"
  [matrix]
  (mapv #(reduce + %) matrix))

(defn sum-cols
  "A vector of the sums of each column in a matrix"
  [matrix]
  (sum-rows (transpose matrix)))

(defn laplace-smooth
  "Add a very small value to each element to prevent division by zero and other ill-effects of zeros in the matrix"
  [matrix alpha]
  (mapv #(mapv + (repeat alpha) %) matrix))

(defn ratio-matrix
  [mat]
  (let [sums (sum-rows mat)]
    (mapv (fn [row sum] (mapv #(/ % (* 1.0 sum)) row)) mat sums)))

(defn apply-ratios
  [row-sums ratio-matrix]
  (mapv (fn [row-sum ratio-vec] (mapv * (repeat row-sum) ratio-vec)) row-sums ratio-matrix))

(defn solve
  [marginals mask]
  (let [ratios (ratio-matrix mask)]
    (apply-ratios marginals ratios)))

(defn back-propogation
  "Send error back through predicted matrix"
  [learning-rate]
  (fn [ys y-hats]
    (let [errors (mapv - ys y-hats)]
      (mapv + y-hats (mapv * (repeat learning-rate) errors)))))

(defn step-once
  [x-marginal y-marginal matrix]
  (let [a (solve x-marginal matrix)
        b (solve y-marginal (transpose a))]
    (transpose b)))

(defn step-10000
  [x-marginal y-marginal matrix]
  (loop [n 10000
         matrix matrix]
    (if (= n 0) matrix
        (recur (dec n) (step-once x-marginal y-marginal matrix)))))

(defn add-labels
  ([header rows]
   (fn [matrix]
     (->> matrix
          (map cons rows)
          (cons header)
          (mapv vec)))))

;;;;; DELETE BELOW WHEN FINISHED ;;;;;

(def visits
  (let [data (open-csv "resources/visits.csv")]
    (cons (first data) (sort (rest data)))))

(def keywords
  (let [data (open-csv "resources/keywords.csv")]
    (remove #(= "0" (last %))
            (cons (first data) (sort (rest data))))))

(def kw-list (mapv first keywords))

(def kw-costs (->> keywords
                   rest
                   (map second)
                   (mapv read-string)
                   (mapv #(* % 0.000001))))

(def hours (remove #(= "0" (last %)) (open-csv "resources/hours.csv")))

(def hour-list (mapv first hours))

(def hour-costs (->> hours
                     rest
                     (map second)
                     (mapv read-string)
                     (mapv #(* % 0.000001))))

(def missing (diff (map first keywords) (map first visits)))

(def visits'
  (let [size (dec (count hours))
        ms   (mapv second missing)]
    (vec (concat visits (mapv #(apply vector % (repeat (inc size) "0")) ms)))))

(def visit-header (first visits'))

(def visit-body (sort-by first (rest visits')))

(def visit-keywords (sort (mapv first visit-body)))

(def visit-matrix
  (->> visit-body
       (map rest)
       (map butlast)
       (map #(map read-string %))
       (mapv vec)))

(def smoothed-visits (laplace-smooth visit-matrix 0.001))

(def solved (step-once kw-costs hour-costs smoothed-visits))

(def solved' (step-once kw-costs hour-costs solved))

(def solved'' (step-once kw-costs hour-costs solved'))

(def solved''' (step-once kw-costs hour-costs solved''))

(def solved'''' (step-once kw-costs hour-costs solved'''))
