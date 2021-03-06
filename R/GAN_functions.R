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
    self$bn <- torch::nn_batch_norm1d(o)
    # Followed by a leakyReLU activation.
    self$leaky_relu <- torch::nn_leaky_relu()
  },
  forward = function(input) {
    # A forward pass will take the input and pass it through the linear layer
    out <- self$fc(input)
    # Then on each element of the output apply the leaky_relu activation
    out <- self$bn(out)
    out <- self$leaky_relu(out)
    # To pass the input through as well we concatenate (cat) the out and input tensor.
    torch::torch_cat(list(out, input), dim = 2)
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
      flow_layers <- replicate(flow_depth, InverseAutoregressiveFlow(data_dim, hidden_size, data_dim), simplify = F) 
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
                                    torch::nn_layer_norm(fc_out_size),
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
    qz_x <- torch::distr_normal(mu_logvar[[1]]$cpu(), std$cpu())
    z <- qz_x$rsample(k_samples)
    log_q_z <- qz_x$log_prob(z)$to(device = device)
    z <- z$to(device = device)
    if(!is.null(self$q_z_flow)) {
      log_q_z_flow <- self$q_z_flow(z, context = fc_out[[3]])
      log_q_z <- log_q_z_flow$sum(-1)
    } else {
      log_q_z <- log_q_z$sum(-1)
    }
    return(list(z, log_q_z))
  },
  encode = function(input, nothing) {
    x <- self$main(input)
    fc_out <- self$fc(x)$chunk(self$enc_chunk, dim = 2)
    mu_logvar <- fc_out[1:2]
    std <- torch::nnf_softplus(mu_logvar[[2]])
    qz_x <- torch::distr_normal(mu_logvar[[1]]$cpu(), std$cpu())
    z <- qz_x$rsample()$to(device = device)
    if(!is.null(self$q_z_flow)){
      z_ <- self$q_z_flow(z, context = fc_out[[3]])
    }
    return(z_[[1]])
  }
)
# And we can define the architecture for the Discriminator as a nn_module.
Discriminator <- torch::nn_module(
  initialize = function(data_dim, # The number of columns in our data
                        hidden_units = list(128, 128), # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5, # The dropout probability
                        pack = 1,
                        sigmoid = FALSE
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
      self$seq$add_module(module = (torch::nn_linear(dim, neurons)),
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
    self$seq$add_module(module = (torch::nn_linear(dim, 1)),
                        name = "Logits")
    
    if(sigmoid){
    self$seq$add_module(module = torch::nn_sigmoid(),
                                            name = "Output")
     }
  },
  calc_gradient_penalty = function(D,real_data, fake_data, device=device, pac=1, lambda_=10){
        alpha = torch::torch_rand(real_data$size(1), 1, device=device)
        alpha = alpha$expand(c(real_data$size(1),real_data$size(2)))
        #alpha = alpha$view(-1, real_data$size(2))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = D(interpolates)

        gradients = torch::autograd_grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch::torch_ones(disc_interpolates$size(), device=device),
            create_graph=TRUE, retain_graph=TRUE
        )[[1]]

        
        
        gradient_penalty = ((gradients$norm(2, dim=2) - 1) ^ 2)$mean() * lambda_
        return(gradient_penalty)
    },
  forward = function(input) {
    data <- self$seq(input$view(c(-1, self$packdim)))
    data
  }
)

Encoder <- torch::nn_module(
  initialize = function(noise_dim,
                        # The length of our noise vector per example
                        data_dim,
                        # The number of columns in our data
                        hidden_units = list(128, 128),
                        # A list with the number of neurons per layer. If you add more elements to the list you create a deeper network.
                        dropout_rate = 0.5 # The dropout probability
                        )
                        {
                        # Initialize an empty nn_sequential module
                        self$seq <- torch::nn_sequential()

                        # For the hidden layers we need to keep track of our input and output dimensions. The first input will be our noise vector, therefore, it will be noise_dim
                        dim <- noise_dim

                        # i will be a simple counter to keep track of our network depth
                        i <- 1

                        # Now we loop over the list of hidden units and add the hidden layers to the nn_sequential module
                        for (neurons in hidden_units) {
                          self$seq$add_module(module = torch::nn_linear(dim, neurons),
                                              name = paste0("Linear_", i))
                          # Add a leakyReLU activation
                          self$seq$add_module(module = torch::nn_leaky_relu(),
                                              name = paste0("Activation_", i))
                          # And a Dropout layer
                          self$seq$add_module(module = torch::nn_dropout(dropout_rate),
                                              name = paste0("Dropout_", i))

                          dim <- neurons
                          # Update the counter
                          i <- i + 1
                        }
                        # Finally, we add the output layer. The output dimension must be the same as our data dimension (data_dim).
                        # self$seq$add_module(module = nn_linear(dim, data_dim),
                        #                     name = "Output")
                        self$enc_mu <- torch::nn_linear(dim, data_dim)
                        self$enc_logvar <- torch::nn_linear(dim, data_dim)
                        },
forward = function(input) {
  output <- self$seq(input)
  mu <- self$enc_mu(output)
  logvar <- self$enc_logvar(output)
  return(list(mu, logvar))
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
    w <- self$module$parameters[[paste0(self$name, "_bar")]]

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

max_singular_value <- function(W, u=NULL, Ip=1){
   
    if(!is.null(u)){
        u = torch::torch_tensor(1, W$size(1))$normal_(0, 1)$to(device)
      }
    bar_u = u
    for(i in 1:Ip){
        bar_v = l2normalize(torch::torch_matmul(bar_u, W$data()), eps=1e-12)
        bar_u = l2normalize(torch::torch_matmul(bar_v, torch::torch_transpose(W$data(), 1, 2)), eps=1e-12)
      }
    sigma = torch::torch_sum(torch::nnf_linear(bar_u, torch::torch_transpose(W$data(), 1, 2)) * bar_v)
    return(list(sigma, bar_u))
}

apply_activate <- function(data, transformer, temperature = 0.2) {
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
      transformed <- torch::nnf_gumbel_softmax(data[,st:ed]$cpu(), tau = temperature)$to(device = device)
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

  return(M_new)
}


sample_g_output <-
  function(encoder, decoder, data, mask, transformer) {
    dim <- data$shape[2]
    data <- data * mask
    torch::with_no_grad({
      encoded_data <- encoder(data$to(device = device))
      decoded_data <-
        apply_activate(decoder(encoded_data), transformer)$detach()$cpu()
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
                  
                  
                  
  sample_g_output2 <-
  function(encoder, decoder, data, mask, transformer) {
    dim <- data$shape[2]
    data <- data * mask
    torch::with_no_grad({
      c(mu_z_enc, var_z_enc) %<-% encoder(data$to(device = device))
      encoded_data <- gauss_repara(mu_z_enc, var_z_enc)
      decoded_data <-
        apply_activate(decoder(encoded_data), transformer)$detach()$cpu()
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



