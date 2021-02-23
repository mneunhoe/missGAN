# The Generator Network will contain so called residual blocks. These pass the output and the input of a layer to the next layer
InverseAutoregressiveFlow <- torch::nn_module(
  initialize = function(num_input, num_hidden, num_context) {
    self$made <- MADE(num_input=num_input, num_output=num_input * 2,
                      num_hidden=num_hidden, num_context=num_context)
    self$sigmoid_arg_bias <- torch::nn_parameter(torch::torch_ones(num_input, device = device) * 2)
    self$sigmoid <- torch::nn_sigmoid()
    self$log_sigmoid <- torch::nn_log_sigmoid()
  },
  forward = function(input, context = NULL) {
    ms <- torch::torch_chunk(self$made(input, context), chunks = 2, dim = -1)
    s <- ms[[2]] + self$sigmoid_arg_bias
    sigmoid <- self$sigmoid(s)
    z <- sigmoid * input + (1 - sigmoid) + ms[[1]]
    
    return(list(z, -self$log_sigmoid(s)))
  }
)




# The Generator Network will contain so called residual blocks. These pass the output and the input of a layer to the next layer
# FlowSequential <- torch::nn_sequential(
#   
#   forward = function(input, context = NULL) {
#     total_log_prob <- torch::torch_zeros_like(input)
#     for(block in self$modules$values){
#       input_log_prob <- block(input, context)
#       total_log_prob <- total_log_prob + log_prob
#     }
#     return(list(input, total_log_prob))
#   }
# )


FlowSequential <- function(... , name = NULL) {
  module <- nn_module(
    classname = ifelse(is.null(name), "flow_sequential", name),
    initialize = function(...) {
      modules <- rlang::list2(...)
      for (i in seq_along(modules)) {
        self$add_module(name = i - 1, module = modules[[i]])  
      }
    },
    forward = function(input, context = NULL) {
      total_log_prob <- torch::torch_zeros_like(input, device = device)
      for(block in self$modules$values){
        input_log_prob <- block(input, context)
        total_log_prob <- total_log_prob + log_prob
      }
      return(list(input, total_log_prob))
    }
  )
  module(...)
}

MaskedLinear <- torch::nn_module(
  initialize = function(in_features, out_features, mask, context_features = NULL, bias = TRUE) {
    self$linear <- torch::nn_linear(in_features, out_features, bias)
    self$register_buffer("mask", mask)
    if(!is.null(context_features)){
      self$cond_linear <- torch::nn_linear(context_features, out_features, bias = FALSE)
    }
  },
  forward = function(input, context = NULL) {
    output <- torch::nnf_linear(input, self$mask * self$linear$weight, self$linear$bias)
    if(is.null(context)) {
      return(output)
    } else {
      return(output + self$cond_linear(context))
    }
  }
)

MADE <- torch::nn_module(
  initialize = function(num_input, num_output, num_hidden, num_context) {
    self$m_ <- list()
    self$masks <- list()
    self$build_masks(num_input, num_output, num_hidden, num_layers = 3)
    self$check_masks()
    modules <- torch::nn_sequential()
    self$input_context_net <- MaskedLinear(num_input, num_hidden, self$masks[[1]], num_context)
    modules$add_module(name = "relu1", module = torch::nn_relu())
    modules$add_module(name = "H1_masked_linear", module = MaskedLinear(
      num_hidden, num_hidden, self$masks[[2]], context_features=NULL))
    modules$add_module(name = "relu2", module = torch::nn_relu())
    modules$add_module(name = "H2_masked_linear", module = MaskedLinear(
      num_hidden, num_hidden, self$masks[[3]], context_features=NULL))
    self$net <- modules
  },
  build_masks = function(num_input, num_output, num_hidden, num_layers) {
    if(!exists(".Random.seed")){
      assign(".Random.seed", NULL, envir = .GlobalEnv)
    }
    old <- .Random.seed
    on.exit( { assign(".Random.seed", old, envir = .GlobalEnv) } )
    set.seed(0)
    self$m_[[length(self$m_)+1]] <- 1:num_input
    for(i in 1:num_layers) {
      if(i == num_layers) {
        m <- 1:num_input
        self$m_[[length(self$m_)+1]] <- do.call("c", replicate(num_output %/% num_input, m, simplify = F))
      } else {
        self$m_[[length(self$m_)+1]] <- sample(1:(num_input-1), num_hidden, replace = T)
      }
      if(i == num_layers) {
        
        a <- array(self$m_[[i]], dim = c(length(self$m_[[i]]), 1))
        b <-  array(self$m_[[i+1]], dim = c(1, length(self$m_[[i+1]])))
        mask <- t(apply(a, 1, function(x) b > x))
        #mask <- t(apply(array(self$m_[[i - 1]], dim = c(length(self$m_[[i-1]]), 1), 1, function(x) array(self$m_[[i]], dim = c(1, length(self$m_[[i]]))) > x)))
        
        #mask = array(self$m_[[i]], dim = c(1, length(self$m_[[i]]))) > array(self$m_[[i - 1]], dim = c(length(self$m_[[i-1]]), 1))
      } else {
        a <- array(self$m_[[i]], dim = c(length(self$m_[[i]]), 1))
        b <-  array(self$m_[[i+1]], dim = c(1, length(self$m_[[i+1]])))
        mask <- t(apply(a, 1, function(x) b >= x))
        #mask <- t(apply(array(self$m_[[i - 1]], dim = c(length(self$m_[[i-1]]), 1), 1, function(x) array(self$m_[[i]], dim = c(1, length(self$m_[[i]]))) >= x))
                  
        #mask = array(self$m_[[i]], dim = c(1, length(self$m_[[i]]))) >= array(self$m_[[i - 1]], dim = c(length(self$m_[[i-1]]), 1))
      }
      self$masks[[length(self$masks)+1]] <- torch::torch_tensor(t(mask*1), device = device)
      
      
    }
    
  },
  check_masks = function() {
    prev <- self$masks[[1]]$t()
    for(i in 2:length(self$masks)) {
      prev <- torch::torch_matmul(prev, self$masks[[i]]$t())
    }
    final <- torch::as_array(prev$cpu())
    num_input <- self$masks[[1]]$shape[2]
    num_output <- self$masks[[length(self$masks)]]$shape[1]
    all(dim(final) == c(num_input, num_output))
    if(num_output == num_input) {
      all(final[upper.tri(final, diag = T)] == 0)
    } else {
      submats <- mat_split(final, nrow(final), ncol(final)/(num_output%/%num_input))
      for(submat in submats) {
        all(submat[upper.tri(submat, diag = T)] == 0)
      }
      
    }
  },
  forward = function(input, context = NULL) {
    hidden <- self$input_context_net(input, context)
    return(self$net(hidden))
  }
)


mat_split <- function(M, r, c){
  nr <- ceiling(nrow(M)/r)
  nc <- ceiling(ncol(M)/c)
  newM <- matrix(NA, nr*r, nc*c)
  newM[1:nrow(M), 1:ncol(M)] <- M
  
  div_k <- kronecker(matrix(seq_len(nr*nc), nr, byrow = TRUE), matrix(1, r, c))
  matlist <- split(newM, div_k)
  N <- length(matlist)
  mats <- unlist(matlist)
  dim(mats)<-c(r, c, N)
  mats <- lapply(seq(dim(mats)[3]), function(x) mats[ , , x])
  return(mats)
}


# From https://stackoverflow.com/questions/14500707/select-along-one-of-n-dimensions-in-array
index_array <- function(x, dim, value, drop = FALSE) {
  # Create list representing arguments supplied to [
  # bquote() creates an object corresponding to a missing argument
  indices <- rep(list(bquote()), length(dim(x)))
  indices[[dim]] <- value
  
  # Generate the call to [
  call <- as.call(c(
    list(as.name("["), quote(x)),
    indices,
    list(drop = drop)))
  # Print it, just to make it easier to see what's going on
  print(call)
  
  # Finally, evaluate it
  eval(call)
}

Reverse <- torch::nn_module(
  initialize = function(num_input) {
    self$perm <- num_input:1
    self$inv_perm <- order(self$perm)
    
  },
  forward = function(inputs, context = NULL, mode = "forward") {
    if(mode == "forward"){
      return(list(index_array(inputs, length(dim(inputs)), self$perm), torch::torch_zeros_like(inputs)))
    } else if(mode == "inverse") {
      return(list(index_array(inputs, length(dim(inputs)), self$inv_perm), torch::torch_zeros_like(inputs)))
    } else {
      stop("Mode must be one of forward or inverse.")
    }
  }
)



