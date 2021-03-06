% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/DataTransformer.R
\name{DataTransformer}
\alias{DataTransformer}
\title{DataTransformer}
\value{
A plot of the toydata
}
\description{
Provides a class to transform data for missGAN
}
\examples{
transformer <- DataTransformer$new()
transformer$fit(data, discrete_columns = 1)
transformed_data <- transformer$transform(data)
retransformed_data <- transformer$inverse_transform(transformed_data)
}
\section{Methods}{
\subsection{Public methods}{
\itemize{
\item \href{#method-new}{\code{DataTransformer$new()}}
\item \href{#method-fit_continuous}{\code{DataTransformer$fit_continuous()}}
\item \href{#method-fit_discrete}{\code{DataTransformer$fit_discrete()}}
\item \href{#method-fit}{\code{DataTransformer$fit()}}
\item \href{#method-transform_continuous}{\code{DataTransformer$transform_continuous()}}
\item \href{#method-transform_discrete}{\code{DataTransformer$transform_discrete()}}
\item \href{#method-transform}{\code{DataTransformer$transform()}}
\item \href{#method-inverse_transform_continuous}{\code{DataTransformer$inverse_transform_continuous()}}
\item \href{#method-inverse_transform_discrete}{\code{DataTransformer$inverse_transform_discrete()}}
\item \href{#method-inverse_transform}{\code{DataTransformer$inverse_transform()}}
\item \href{#method-clone}{\code{DataTransformer$clone()}}
}
}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-new"></a>}}
\if{latex}{\out{\hypertarget{method-new}{}}}
\subsection{Method \code{new()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$new()}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-fit_continuous"></a>}}
\if{latex}{\out{\hypertarget{method-fit_continuous}{}}}
\subsection{Method \code{fit_continuous()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$fit_continuous(column = NULL, data = NULL)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-fit_discrete"></a>}}
\if{latex}{\out{\hypertarget{method-fit_discrete}{}}}
\subsection{Method \code{fit_discrete()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$fit_discrete(column = NULL, data = NULL)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-fit"></a>}}
\if{latex}{\out{\hypertarget{method-fit}{}}}
\subsection{Method \code{fit()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$fit(data, discrete_columns = NULL)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-transform_continuous"></a>}}
\if{latex}{\out{\hypertarget{method-transform_continuous}{}}}
\subsection{Method \code{transform_continuous()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$transform_continuous(column_meta, data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-transform_discrete"></a>}}
\if{latex}{\out{\hypertarget{method-transform_discrete}{}}}
\subsection{Method \code{transform_discrete()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$transform_discrete(column_meta, data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-transform"></a>}}
\if{latex}{\out{\hypertarget{method-transform}{}}}
\subsection{Method \code{transform()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$transform(data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-inverse_transform_continuous"></a>}}
\if{latex}{\out{\hypertarget{method-inverse_transform_continuous}{}}}
\subsection{Method \code{inverse_transform_continuous()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$inverse_transform_continuous(meta, data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-inverse_transform_discrete"></a>}}
\if{latex}{\out{\hypertarget{method-inverse_transform_discrete}{}}}
\subsection{Method \code{inverse_transform_discrete()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$inverse_transform_discrete(meta, data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-inverse_transform"></a>}}
\if{latex}{\out{\hypertarget{method-inverse_transform}{}}}
\subsection{Method \code{inverse_transform()}}{
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$inverse_transform(data)}\if{html}{\out{</div>}}
}

}
\if{html}{\out{<hr>}}
\if{html}{\out{<a id="method-clone"></a>}}
\if{latex}{\out{\hypertarget{method-clone}{}}}
\subsection{Method \code{clone()}}{
The objects of this class are cloneable with this method.
\subsection{Usage}{
\if{html}{\out{<div class="r">}}\preformatted{DataTransformer$clone(deep = FALSE)}\if{html}{\out{</div>}}
}

\subsection{Arguments}{
\if{html}{\out{<div class="arguments">}}
\describe{
\item{\code{deep}}{Whether to make a deep clone.}
}
\if{html}{\out{</div>}}
}
}
}
