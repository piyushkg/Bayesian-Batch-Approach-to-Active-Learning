data {
  int N; // No. of training observations n=N, k=K, p=M
  int M; // No. of features
  int K; // No. of Classes
  int Y[N,K]; // the training output
  matrix[N,M] X; // Model Matrix
}

transformed data {
  int flattened_y[N * K] = to_array_1d(Y);
}
parameters {
  matrix[M,K] beta; //the regression parameters
  //matrix[p, k] beta_1; // species response to predictors
}
model {
  matrix[N,K] X_beta = X * beta;
  to_vector(beta) ~ normal(0, 50);
  
  flattened_y ~ bernoulli_logit(to_vector(X_beta'));
  
}
