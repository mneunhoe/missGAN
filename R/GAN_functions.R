# First, we check whether a compatible GPU is available for computation.
use_cuda <- torch::cuda_is_available()
#use_cuda <- F
# If so we would use it to speed up training.
device <- ifelse(use_cuda, "cuda", "cpu")

# The Generator Network will contain so called residual blocks. These pass the output and the input of a layer to the next layer
ResidualBlock <- torch::nn_module(
  initialize = function(i, o) {
    # We will use a fully connected (fc) linear layer
    self$fc <- torch::nn_linear(i, o)
    # Followed by a leakyReLU activation.
    self$leaky_relu <- torch::nn_leaky_relu()
  },
  forward = function(input) {
    # A forward pass will take the input and pass it through the linear layer
    out <- self$fc(input)
    # Then on each element of the output apply the leaky_relu activation
    out <- self$leaky_relu(out)
    # To pass the input through as well we concatenate (cat) the out and input tensor.
    torch::torch_cat(list(out, input), dim = 2)
  }
)

# Now we can define the architecture for the Generator as a nn_module.
FlowEncoder <- torch::nn_module(
  initialize = function(noise_dim, # The length of our noise vector per example
                        data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        flow_depth = 2,
                        logprob = FALSE,
                        dropout_rate = 0 # The dropout probability
  ) {

    if(logprob) {
      self$encode_func <- self$encode_logprob
    } else
      self$encode_func <- self$encode

    dim <- noise_dim
    # Initialize an empty nn_sequential module
    self$main <- torch::nn_sequential()

    # i will be a simple counter to keep track of our network depth
    i <- 1

    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # First, we add a ResidualBlock of the respective size.
      self$main$add_module(module =  ResidualBlock(dim, neurons),
                          name = paste0("ResBlock_", i))
      # And then a Dropout layer.
      self$main$add_module(module = torch::nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Now we update our dim for the next hidden layer.
      # Since it will be another ResidualBlock the input dimension will be dim+neurons
      dim <- dim + neurons
      # Update the counter
      i <- i + 1
    }
    # Finally, we add the output layer. The output dimension must be the same as our data dimension (data_dim).
    self$main$add_module(module = torch::nn_linear(dim, 4*dim),
                        name = "Output")

    if(flow_depth > 0) {
      hidden_size <- data_dim * 2
      flow_layers <- replicate(flow_depth, InverseAutoregressiveFlow(data_dim, hidden_size, latent_size), simplify = F)
      flow_layers[[length(flow_layers)+1]] <- Reverse(data_dim)
      self$q_z_flow <- do.call(FlowSequential, flow_layers)
      self$enc_chunk <- 3
    } else {
      self$q_z_flow <- NULL
      self$enc_chunk <- 2
    }
    fc_out_size <- data_dim * self$enc_chunk
    out_size <- 4*dim
    self$fc <- torch::nn_sequential(torch::nn_linear(out_size, fc_out_size),
                                    torch::nnf_layer_norm(fc_out_size),
                                    torch::nn_leaky_relu(0.2),
                                    torch::nn_linear(fc_out_size, fc_out_size))
  },
  forward = function(input, k_samples = 5) {
    self$encode_func(input, k_samples)
  },
  encode_logprob = function(input, k_samples = 5) {
    x <- self$main(input)
    fc_out <- self$fc(x)$chunk(self$enc_chunk, dim = 2)
    mu_logvar <- fc_out[1:2]
    std <- torch::nnf_softplus(mu_logvar[[2]])
    return(NULL)
  },
  encode = function(input, nothing) {
    x <- self$main(input)
    fc_out <- self$fc(x)$chunk(self$enc_chunk, dim = 2)
    mu_logvar <- fc_out[1:2]
    std <- torch::nnf_softplus(mu_logvar[[2]])
    z <- torch::torch_tensor(rnorm(n = length(torch::as_array(mu)) , torch::as_array(mu), torch::as_array(std)))
    if(self$q_z_flow){
      z_ <- self$q_z_flow(z, context = fc_out[[3]])
    }
    return(z_[[1]])
  }
)



# Now we can define the architecture for the Generator as a nn_module.
Generator <- torch::nn_module(
  initialize = function(noise_dim, # The length of our noise vector per example
                        data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5 # The dropout probability
  ) {
    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- noise_dim

    # i will be a simple counter to keep track of our network depth
    i <- 1

    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # First, we add a ResidualBlock of the respective size.
      self$seq$add_module(module =  ResidualBlock(dim, neurons),
                          name = paste0("ResBlock_", i))
      # And then a Dropout layer.
      self$seq$add_module(module = torch::nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Now we update our dim for the next hidden layer.
      # Since it will be another ResidualBlock the input dimension will be dim+neurons
      dim <- dim + neurons
      # Update the counter
      i <- i + 1
    }
    # Finally, we add the output layer. The output dimension must be the same as our data dimension (data_dim).
    self$seq$add_module(module = torch::nn_linear(dim, data_dim),
                        name = "Output")
  },
  forward = function(input) {
    input <- self$seq(input)
    input
  }
)


# And we can define the architecture for the Discriminator as a nn_module.
Discriminator <- torch::nn_module(
  initialize = function(data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5, # The dropout probability
                        pack = 1
  ) {

    # Initialize an empty nn_sequential module
    self$seq <- torch::nn_sequential()

    # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
    dim <- data_dim * pack
    self$pack <- pack
    self$packdim <- dim
    # i will be a simple counter to keep track of our network depth
    i <- 1

    # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
    for (neurons in hidden_units) {
      # We start with a fully connected linear layer
      self$seq$add_module(module = SpectralNorm(torch::nn_linear(dim, neurons)),
                          name = paste0("Linear_", i))
      # Add a leakyReLU activation
      self$seq$add_module(module = torch::nn_leaky_relu(),
                          name = paste0("Activation_", i))
      # And a Dropout layer
      self$seq$add_module(module = torch::nn_dropout(dropout_rate),
                          name = paste0("Dropout_", i))
      # Update the input dimension to the next layer
      dim <- neurons
      # Update the counter
      i <- i + 1
    }
    # Add an output layer to the net. Since it will be one score for each example we only need a dimension of 1.
    self$seq$add_module(module = SpectralNorm(torch::nn_linear(dim, 1)),
                        name = "Output")

  },
  forward = function(input) {
    data <- self$seq(input$view(c(-1, self$packdim)))
    data
  }
)


SpectralNorm <- torch::nn_module(
  initialize = function(module, name = "weight", power_iterations = 1) {
    self$module <- module
    self$name <- name
    self$power_iterations <- power_iterations
    if(!self$made_params()){
      self$make_params()
    }
  },
  update_u_v = function() {
    u <- self$module$parameters[[paste0(self$name, "_u")]]
    v <- self$module$parameters[[paste0(self$name, "_v")]]
    w <- self$module$parameters[[paste0(self$name)]]

    height <- w$data()$shape[1]
    for(i in 1:self$power_iterations) {
      v <- l2normalize(torch::torch_mv(torch::torch_t(w$view(c(height, -1))$data()), u$data()))
      u <- l2normalize(torch::torch_mv(w$view(c(height, -1))$data(), v$data()))
    }
    sigma <- u$dot(w$view(c(height, -1))$mv(v))
    self$module[[paste0(self$name)]] <- w / sigma$expand_as(w)

  },
  made_params = function() {
    ifelse(all(paste0(self$name, c("_u", "_v", "_bar")) %in% names(self$module$parameters)), TRUE, FALSE)
  },
  make_params = function() {
    w <- self$module$parameters[[paste0(self$name)]]

    height <- w$data()$shape[1]
    width <- w$view(c(height, -1))$data()$shape[1]

    u <- torch::nn_parameter(l2normalize(torch::torch_randn(height)), requires_grad = FALSE)
    v <- torch::nn_parameter(l2normalize(torch::torch_randn(width)), requires_grad = FALSE)

    # u <- l2normalize(u$data())
    # v <- l2normalize(v$data())
    w_bar <- torch::nn_parameter(w$data())

    self$module$register_parameter(paste0(self$name, "_u"), u)
    self$module$register_parameter(paste0(self$name, "_v"), v)
    self$module$register_parameter(paste0(self$name, "_bar"), w_bar)
  },
  forward = function(...) {
    self$update_u_v()
    return(self$module$forward(...))

  }
)



# The Generator Network will contain so called residual blocks. These pass the output and the input of a layer to the next layer
InverseAutoregressiveFlow <- torch::nn_module(
  initialize = function(num_input, num_hidden, num_context) {
    self$made <- MADE(num_input=num_input, num_output=num_input * 2,
                      num_hidden=num_hidden, num_context=num_context)
    self$sigmoid_arg_bias <- torch::nn_parameter(torch::torch_ones(num_input) * 2)
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
FlowSequential <- torch::nn_sequential(

  forward = function(input, context = NULL) {
    total_log_prob <- torch::torch_zeros_like(input)
    for(block in self$modules$values){
      input_log_prob <- block(input, context)
      total_log_prob <- total_log_prob + log_prob
    }
    return(list(input, total_log_prob))
  }
)

MaskedLinear <- torch::nn_module(
  initialize = function(in_features, out_features, mask, context_features = NULL, bias = TRUE) {
    self$linear <- torch::nn_linear(in_features, out_features, bis)
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
    self$m <- list()
    self$masks <- list()
    self$build_masks(num_input, num_output, num_hidden, num_layers = 3)
    self$check_masks()
    modules <- torch::nn_sequential()
    self$input_context_net <- MaskedLinear(num_input, num_hidden, self$masks[[1]], num_context)
    modules$add_module(module = torch::nn_relu())
    modules$add_module(module = MaskedLinear(
      num_hidden, num_hidden, self$masks[[2]], context_features=NULL))
    modules$add_module(module = torch::nn_relu())
    modules$add_module(module = MaskedLinear(
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
    self$m[[length(self$m)+1]] <- 1:num_input
    for(i in 2:num_layers) {
      if(i == num_layers) {
        m <- 1:num_input
        self$m[[length(self$m)+1]] <- do.call("c", replicate(num_output %/% num_input, m, simplify = F))
      } else {
        self$m[[length(self$m)+1]] <- sample(1:(num_input-1), num_hidden, replace = T)
      }
      if(i == num_layers) {
        mask = array(self$m[[i]], dim = c(1, length(self$m[[i]]))) > array(self$m[[i - 1]], dim = c(length(self$m[[i-1]]), 1))
      } else {
        mask = array(self$m[[i]], dim = c(1, length(self$m[[i]]))) >= array(self$m[[i - 1]], dim = c(length(self$m[[i-1]]), 1))
      }
      self$masks[[length(self$masks)+1]] <- torch::torch_tensor(t(mask*1))


    }

  },
  check_masks = function() {
    prev <- self$masks[[1]]$t()
    for(i in 2:length(self$masks)) {
      prev <- torch::torch_matmul(prev, self$masks[[i]]$t())
    }
    final <- torch::as_array(prev)
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
    }
  } else {
    stop("Mode must be one of forward or inverse.")
  }
)

# We will use the kl GAN loss
# You can find the paper here: https://arxiv.org/abs/1910.09779
# And the original python implementation here: https://github.com/ermongroup/f-wgan

kl_real <- function(dis_real) {
  loss_real <- torch::torch_mean(torch::nnf_relu(1 - dis_real))

  return(loss_real)
}

kl_fake <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss_fake = torch::torch_mean(torch::nnf_relu(1. + dis_fake))

  return(loss_fake)
}

kl_gen <- function(dis_fake) {
  dis_fake_norm = torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio = torch::torch_exp(dis_fake) / dis_fake_norm
  dis_fake = dis_fake * dis_fake_ratio
  loss = -torch::torch_mean(dis_fake)
  return(loss)
}


hinge_real <- function(dis_real){
  loss_real <- torch::torch_mean(torch::nnf_relu(1. - dis_real))
  return(loss_real)
}



hinge_fake <- function(dis_fake){
  loss_fake <- torch::torch_mean(torch::nnf_relu(1. + dis_fake))
  return(loss_fake)
}


hinge_gen <- function(dis_fake){
  loss <- -torch::torch_mean(dis_fake)
  return(loss)
}


kl_r <- function(dis_fake) {
  dis_fake_norm <- torch::torch_exp(dis_fake)$mean()
  dis_fake_ratio <- torch::torch_exp(dis_fake) / dis_fake_norm
  res <-
    (dis_fake_ratio / dis_fake_norm) * (
      dis_fake - torch::torch_logsumexp(dis_fake, dim = 1) +
        torch::torch_log(torch::torch_tensor(dis_fake$size(1)))$to(device)
    )
  return(res)

}

#define _l2normalization
l2normalize <- function(v, eps=1e-12) {
  return(v / (torch::torch_norm(v) + eps))
}


apply_activate <- function(data, transformer, temperature = .66) {
  DIM <- data$shape[2]
  data_t <- list()
  st <- 1
  for(item in transformer$output_info) {
    if(item[[2]] == "linear"){
      ed <- st + item[[1]] - 1
      data_t[[length(data_t)+1]] <- data[,st:ed]
      st <- ed + 1
    } else if(item[[2]] == "tanh") {
      ed <- st + item[[1]] - 1
      data_t[[length(data_t)+1]] <- torch::torch_tanh(data[,st:ed])
      st <- ed + 1
    } else if(item[[2]] == "softmax") {
      ed <- st + item[[1]] - 1
      transformed <- torch::nnf_gumbel_softmax(data[,st:ed], tau = 0.2)
      data_t[[length(data_t)+1]] <- transformed
      st <- ed + 1
    } else {
      NULL
    }
  }
  data_t[[length(data_t)+1]] <- torch::torch_sigmoid(data[,st:DIM]/temperature)

  return(torch::torch_cat(data_t, dim = 2))

}

apply_mask_activate <- function(data, temperature = 0.66) {
  DIM <- data$shape[2]
  data_t <- torch::torch_sigmoid(data[,1:DIM]/temperature)

  return(data_t)
}

match_mask <- function(M, transformer) {
  end_idxs <-
    cumsum(sapply(transformer$meta, function(x)
      x$output_dimensions))

  start_idxs <- (c(0, end_idxs) + 1)[1:length(end_idxs)]

  idxs_mat <- cbind(start_idxs, end_idxs)

  M_new <- list()



  for (i in 1:nrow(idxs_mat)) {
    n <- length(seq(idxs_mat[i, 1], idxs_mat[i, 2]))
    M_new[[i]] <- replicate(n, M[, i])
  }

  M_new <- do.call("cbind", M_new)

  return(torch::torch_tensor(M_new))
}


sample_g_output <-
  function(encoder, decoder, data, mask, transformer) {
    dim <- data$shape[2]
    data <- data * mask
    torch::with_no_grad({
      encoded_data <- encoder(data$to(device = device))
      decoded_data <-
        apply_activate(decoder(encoded_data))$detach()$cpu()
    })
    synth_data <- data * mask + (1 - mask) * decoded_data
    synth_data <- torch::as_array(synth_data$detach()$cpu())
    synth_data <- transformer$inverse_transform(synth_data)

    decoded_data <-
      transformer$inverse_transform(torch::as_array(decoded_data$detach()$cpu()))

    return(list(encoded_data = torch::as_array(encoded_data$detach()$cpu()),
                imputed_data = synth_data,
                synthetic_data = decoded_data))
  }




