reweight <- function(fam,weights){
    famname <-fam$name
    a <- attributes(fam)
    n <- length(a[["_weights"]])
if(length(weights)!=n)stop("replacement weight vector has to have same length as weight vector on object")
    weights <- weights/sum(weights)
    switch(famname,
           "gaussian" = glm.gaussian(
               y = a[["_y"]],
               weights = weights,
               opt = a[["_opt"]]
           ),
           "binomial_logit" = glm.binomial(
               y = a[["_y"]],
               weights = weights,
               link = "logit"
           ),
           "binomial_probit" = glm.binomial(
               y = a[["_y"]],
               weights = weights,
               link = "probit"
           ),
           "cox" = glm.cox(
               start = a[["_start"]],
               stop = a[["_stop"]],
               status = a[["_status"]],
               weights = weights,
               tie_method = a[["_tie_method"]]
           ),
           "multigaussian"= glm.multigaussian(
               y = a[["_y"]],
               weights = weights,
               opt = a[["_opt"]]
           ),
           "multinomial" = glm.multinomial(
               y = a[["_y"]],
               weights = weights
           ),
           "poisson" = glm.poisson(
               y = a[["_y"]],
               weights = weights
           )
           )
}
print.grpnet <- function (x, digits = max(3, getOption("digits") - 3), ...)
{
    cat("\nCall: ", deparse(x$call), "\n\n")
    coefstuff <- coef(x)
    Df = coefstuff$df
    dev.ratio = x$state$devs
    lambdas=coefstuff$lambda
    out = data.frame(Df, `%Dev` = round(dev.ratio *
        100, 2), Lambda = signif(lambdas, digits), check.names = FALSE,
        row.names = seq(along = Df))
    class(out) = c("anova", class(out))
    print(out)
}


#' make predictions from a "grpnet" object.
#'
#' Similar to other predict methods, this functions predicts linear predictors,
#' coefficients and more from a fitted \code{"grpnet"} object.
#'
#' The shape of the objects returned are different for \code{"multinomial"} and \code{"multigaussian"}
#' objects
#' \code{coef(...)} is equivalent to \code{predict(type="coefficients",...)}
#'
#' @aliases coef.grpnet predict.grpnet
#' @param object Fitted \code{"grpnet"} model object or a \code{"relaxed"}
#' model (which inherits from class "grpnet").
#' @param newx Matrix of new values for \code{x} at which predictions are to be
#' made. Can be a matrix, a sparse matrix as in \code{Matrix} package, or else any of the matrix forms allowable in the \code{adelie} package. This
#' argument is not used for \code{type="coefficients"}
#' @param lambda Value(s) of the penalty parameter \code{lambda} at which
#' predictions are required. Default is the entire sequence used to create the
#' model. If values of \code{lambda} are supplied, the function uses linear
#' interpolation to make predictions for values of \code{lambda} that do
#' not coincide with those used in the fitting algorithm.
#' @param type Type of prediction required. Type \code{"link"} is  the default, and gives the linear
#' predictors. Type \code{"response"} applies the inverse link to these predictions.
#' Type \code{"coefficients"} extracts the coefficients, intercepts and the active-set sizes.
#' @param newoffset If an offset is used in the fit, then one must be supplied
#' for making predictions (except for \code{type="coefficients"} or
#' \code{type="nonzero"})
#' @param \dots Currently ignored.
#' @return The object returned depends on type.
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie
#' \email{hastie@stanford.edu}
#' @seealso \code{grpnet}, and \code{print}, and \code{coef} methods, and
#' \code{cv.grpnet}.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.\cr
#' Adelie Python user guide  \url{https://jamesyang007.github.io/adelie/}
#' @keywords models regression
#'
#' @examples
#' set.seed(0)
#' n <- 100
#' p <- 200
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X[,1] * rnorm(1) + rnorm(n)
#' fit <- grpnet(X, glm.gaussian(y))
#' coef(fit)
#' predict(fit,newx = X[1:5])
#' @method predict grpnet
#' @export
#' @export predict.grpnet
predict.grpnet=function(object,newx,lambda=NULL,type=c("link","response","coefficients"),newoffset,...){
 type=match.arg(type)
  if(missing(newx)){
    if(!match(type,c("coefficients"),FALSE))stop("You need to supply a value for 'newx'")
  }
 if(type=="response")stop("type response not yet implemented")

state <- object$state

 ## Check for multi fit
 is.multi <- FALSE
 if(!is.null(intercepts <-  state[["intercepts_multi"]])){
     ## it is a list
     K <- length(intercepts[[1]])
     intercepts <- unlist(intercepts)
     if(K>1){
         intercepts=matrix(unlist(intercepts),byrow=TRUE,ncol=K)
         is.multi <- TRUE
     }
 } else intercepts <- as.matrix(state$intercepts)
 betas = if(is.multi)state$betas_multi else state$betas
 if(!is.null(lambda)){
     lambda.orig=state$lmdas
     lamlist=lambda.interp(lambda.orig,lambda)
     betas = Diagonal(x=lamlist$frac)%*%betas[lamlist$left,,drop=FALSE] + Diagonal(x=1-lamlist$frac)%*%betas[lamlist$right,,drop=FALSE]
     intercepts = diag(x=lamlist$frac)%*%intercepts[lamlist$left,,drop=FALSE] + diag(x=1-lamlist$frac)%*%intercepts[lamlist$right,,drop=FALSE]
     if(!inherits(betas,"dgRMatrix"))betas = as(betas,"RsparseMatrix")
 } else lambda = state$lmdas
 dof <- diff(betas@p)
 if(is.multi)dof=dof/K
 nlams = nrow(intercepts)
 if(type=="coefficients") return(list(intercepts=intercepts,betas=betas,df=dof,lambda=lambda))

 ## Convert newx to an adelie matrix
 if(inherits(newx,"sparseMatrix")){
     newx <- as(newx,"CsparseMatrix")
     newx <- matrix.sparse(newx, method="naive", n_threads=n_threads)
 }
 if (is.matrix(newx) || is.array(newx) || is.data.frame(newx)) {
     newx <- matrix.dense(newx, method="naive")
 }
 n <- newx$rows

### Now we produce either a prediction matrix (single response), or a prediction array (multi response)
 if(!is.multi){# single target
     preds = newx$sp_btmul(betas)+outer(rep(1,n),drop(intercepts))
     if(!missing(newoffset)){
         if(length(newoffset)!=n)stop("Newoffset should have same number of elements as rows of newx")
         preds=preds+matrix(newoffset,n,nlams)
         }
 }
 else{# multi targets
     newx = matrix.kronecker_eye(newx,K=K)
     preds = newx$sp_btmul(betas)
     nlams = ncol(preds)
     intercepts = matrix(intercepts,nlams,n*K)# recycles
     preds=preds+t(intercepts)
     if(!missing(newoffset)){
         if(!is.matrix(newoffset)||!all.equal(dim(newoffset),c(n,K)))
             stop("Newoffset should be an N x K matrix where K is the number of responses")
         preds=preds+matrix(t(newoffset),n*K,nlams)
     }
     preds = array(preds,c(K,n,nlams))
     preds = aperm(preds,c(2,1,3))
 }
 return(preds)
}


lambda.interp=function(lambda,s){
### lambda is the index sequence that is produced by the model
### s is the new vector at which evaluations are required.
### the value is a vector of left and right indices, and a vector of fractions.
### the new values are interpolated bewteen the two using the fraction
### Note: lambda decreases. you take:
### sfrac*left+(1-sfrac*right)

  if(length(lambda)==1){# degenerate case of only one lambda
    nums=length(s)
    left=rep(1,nums)
    right=left
    sfrac=rep(1,nums)
  }
  else{
      ## s[s > max(lambda)] = max(lambda)
      ## s[s < min(lambda)] = min(lambda)
      k=length(lambda)
      sfrac <- (lambda[1]-s)/(lambda[1] - lambda[k])
      lambda <- (lambda[1] - lambda)/(lambda[1] - lambda[k])
      sfrac[sfrac < min(lambda)] <- min(lambda)
      sfrac[sfrac > max(lambda)] <- max(lambda)
      coord <- approx(lambda, seq(lambda), sfrac)$y
      left <- floor(coord)
      right <- ceiling(coord)
      sfrac=(sfrac-lambda[right])/(lambda[left] - lambda[right])
      sfrac[left==right]=1
      sfrac[abs(lambda[left]-lambda[right])<.Machine$double.eps]=1

    }
list(left=left,right=right,frac=sfrac)
}

#' Extract coefficients from a grpnet object
#'
#' @method coef grpnet
#' @rdname predict.grpnet
#' @export
#' @export coef.grpnet
coef.grpnet=function(object,s=NULL,...)
  predict(object,s=s,type="coefficients",...)






