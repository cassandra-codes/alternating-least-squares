(ns cost-allocation.core
  (:require [cost-allocation.data :refer :all]
            [clojure.math.numeric-tower :refer [round, abs, expt]]))

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

(defn rmse
  [observed predicted]
  (L2-norm (diff-vector observed predicted)))

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

(defn weight-matrix
  "Convert each element in a matrix to a value representing that element's proportion of the sum of its row."
  [mat]
  (let [sums (sum-rows mat)]
    (mapv (fn [row sum] (mapv #(/ % (* 1.0 sum)) row)) mat sums)))

(defn apply-weights
  "Distributes the weights in the `weight-matrix` across a rows summing to the `row-sums` vector" 
  [row-sums weight-matrix]
  (mapv (fn [row-sum weight-vec] (mapv * (repeat row-sum) weight-vec)) row-sums weight-matrix))

(defn add-labels
  ([header rows]
   (fn [matrix]
     (->> matrix
          (map cons rows)
          (cons header)
          (mapv vec)))))

(defn mmult
  "Multiply an n x m matrix by an m x p matrix to produce an n x p matrix."
  [xss yss]
  (let [dot (fn [xs]
              (fn [ys] (reduce + (mapv * xs ys))))]
    (mapv (fn [xs] (mapv (dot xs) (transpose yss))) xss)))

(defn seed-matrix
  [xs ys]
  (mmult (transpose [xs]) [ys]))

(defn optimize-marginal
  "Distribute the values in a `marginal`` across their respective rows/columns so that it's proportional to the distributions in the `seed` matrix."
  [marginal seed]
  (let [ratios (ratio-matrix seed)]
    (apply-ratios marginal ratios)))

(defn step
  "Optimise the `x-marginal` followed by the `y-marginal` once, representing a single iteration towards convergence.
  The `seed` is a matrix representing the rough shape of the solution matrix we are converging towards.  It could effectively be thought of as a loose training label, as the values in the output matrix will be different from the `seed`'s, but the ratios will be quite similar."
  [x-marginal y-marginal seed]
  (let [a (optimize-marginal x-marginal seed)
        b (optimize-marginal y-marginal (transpose a))]
    (transpose b)))

(defn converge
  "Continuously optimise alternating marginals until one of three criteria is met."
  [x-marginal y-marginal seed & {:keys [threshold max-iter verbose?]
                                 :or   {threshold 1e-15
                                        max-iter  1000
                                        verbose?  false}}]
  {:pre [(number? threshold) (int? max-iter) (boolean? verbose?)]}
  (let [normalized-seed (normalize-matrix seed)]
    (loop [matrix    seed
           last-loss Integer/MAX_VALUE
           loss-diff Integer/MAX_VALUE
           iteration max-iter]
      (if (or (< last-loss threshold) ;; stop if our loss function reaches an acceptable threshold
              (< loss-diff threshold) ;; stop if our loss function has hit a local minimum
              (= iteration 0))        ;; stop if it's taking too long.
        {:matrix matrix, :loss last-loss, :iterations (- max-iter iteration)}
        (let* [next (step x-marginal y-marginal matrix)
               loss (rmse x-marginal (sum-rows matrix))
               diff (abs (- last-loss loss))]
          (if verbose? (println {:loss last-loss, :diff loss-diff, :iter iteration}))
          (recur next loss diff (dec iteration)))))))

(defn normalize-matrix
  "Bound every element in a matrix to a value between 0 and 1, such that the largest element in the matrix becomes one, the smallest element becomes zero, and all elements in between are a proportional value of the largest element."
  [matrix]
  (let* [flattened (flatten matrix)
         min-value (apply min flattened)
         max-value (- (apply max flattened) min-value)]
    (mapv (fn [xs] (mapv (fn [x] (/ (- x min-value) max-value)) xs)) matrix)))

(defn error-matrix
  [pss qss]
  (let [diff (fn [p q] (abs (- p q)))]
    (mapv (fn [ps qs] (mapv diff ps qs)) pss qss)))

(defn matrix-norm
  [matrix]
  (reduce + (mapv L2-norm matrix)))

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

(def smoothed-visits (laplace-smooth visit-matrix 1e-15))

(def dummy-visits (mapv (fn [_] (vec (repeat 22 1))) (repeat 73 1)))

(def cost-per-visit3
  (mapv
   (fn
     [xs ys]
     (mapv (fn [x y] (if (zero? y) 0 (/ x y))) xs ys))
   cost-per-visit2
   visit-matrix))
