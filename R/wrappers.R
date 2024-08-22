## Utility function for allowing the weights to be changed in a glm family function.
## This is used in cross-validation, when the model is refit with weights set to zero
## for the observations in the left-out fold.
reweight <- function(fam,weights){
    famname <-fam$name
    a <- attributes(fam)
    n <- length(a[["_weights"]])
if(length(weights)!=n)stop("replacement weight vector has to have same length as weight vector on object")
    weights <- weights/sum(weights)
    switch(famname,
           "gaussian" = glm.gaussian(
               y = fam$y,
               weights = weights,
               opt = a[["opt"]]
           ),
           "binomial_logit" = glm.binomial(
               y = fam$y,
               weights = weights,
               link = "logit"
           ),
           "binomial_probit" = glm.binomial(
               y = fam$y,
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
               y = fam$y,
               weights = weights,
               opt = a[["opt"]]
           ),
           "multinomial" = glm.multinomial(
               y = fam$y,
               weights = weights
           ),
           "poisson" = glm.poisson(
               y = fam$y,
               weights = weights
           )
           )
}

#' print a grpnet object
#'
#' Print a summary of the grpnet path at each step along the path.
#' @details
#' The call that produced the object `x` is printed, followed by a
#' three-column matrix with columns `Df`, `%Dev` and `Lambda`.
#' The `Df` column is the number of nonzero coefficients (Df is a
#' reasonable name only for lasso fits). `%Dev` is the percent deviance
#' explained (relative to the null deviance).
#'
#' @param x fitted grpnet object
#' @param digits significant digits in printout
#' @param \dots additional print arguments
#' @return The matrix above is silently returned
#' @seealso \code{grpnet}, \code{predict}, \code{plot} and \code{coef} methods.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.
#' @keywords models regression
#' @examples
#'
#' x = matrix(rnorm(100 * 20), 100, 20)
#' y = rnorm(100)
#' fit1 = grpnet(x, glm.gaussian(y))
#' print(fit1)
#' @method print grpnet
#' @export

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
#' @param object Fitted \code{"grpnet"} model.
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
#' @param newoffsets If an offset is used in the fit, then one must be supplied
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
predict.grpnet=function(object,newx,lambda=NULL,type=c("link","response","coefficients"),newoffsets=NULL,...){
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
     nlam=length(lambda)
     intercepts = diag(x=lamlist$frac,nrow=nlam)%*%intercepts[lamlist$left,,drop=FALSE] +
         diag(x=1-lamlist$frac,nrow=nlam)%*%intercepts[lamlist$right,,drop=FALSE]
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
     if(!is.null(newoffsets)){
         if(length(newoffsets)!=n)stop("Argument newoffsets should have same number of elements as rows of newx")
         preds=preds+matrix(newoffsets,n,nlams)
         }
 }
 else{# multi targets
     newx = matrix.kronecker_eye(newx,K=K)
     preds = newx$sp_btmul(betas)
     nlams = ncol(preds)
     intercepts = matrix(intercepts,nlams,n*K)# recycles
     preds=preds+t(intercepts)
     if(!is.null(newoffsets)){
         if(!is.matrix(newoffsets)||!all.equal(dim(newoffsets),c(n,K)))
             stop("Argument newoffsets should be an n x K matrix where n is the number of rows of newx and K is the number of responses")
         preds=preds+matrix(t(newoffsets),n*K,nlams)
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


#' Cross-validation for grpnet
#'
#' Does k-fold cross-validation for grpnet
#'
#' The function runs \code{grpnet} \code{n_folds}+1 times; the first to get the
#' \code{lambda} sequence, and then the remainder to compute the fit with each
#' of the folds omitted. The out-of-fold deviance is accumulated, and the average deviance and
#' standard deviation over the folds is computed.  Note that \code{cv.grpnet}
#' does NOT search for values for \code{alpha}. A specific value should be
#' supplied, else \code{alpha=1} is assumed by default. If users would like to
#' cross-validate \code{alpha} as well, they should call \code{cv.grpnet} with
#' a pre-computed vector \code{foldid}, and then use this same \code{foldid} vector in
#' separate calls to \code{cv.grpnet} with different values of \code{alpha}.
#' Note also that the results of \code{cv.grpnet} are random, since the folds
#' are selected at random. Users can reduce this randomness by running
#' \code{cv.grpnet} many times, and averaging the error curves.
#'
#' @param X Feature matrix. Either a regualr R matrix, or else an
#'     \code{adelie} custom matrix class, or a concatination of such.
#' @param glm GLM family/response object. This is an expression that
#'     represents the family, the reponse and other arguments such as
#'     weights, if present. The choices are \code{glm.gaussian()},
#'     \code{glm.binomial()}, \code{glm.poisson()},
#'     \code{glm.multinomial()}, \code{glm.cox()}, \code{glm.multinomial()},
#'     and \code{glm.multigaussian()}. This is a required argument, and
#'     there is no default. In the simple example below, we use \code{glm.gaussian(y)}.
#' @param n_folds (default 10). Although \code{n_folds} can be
#' as large as the sample size (leave-one-out CV), it is not recommended for
#' large datasets. Smallest value allowable is \code{n_folds=3}.
#' @param foldid An optional vector of values between 1 and \code{n_folds}
#' identifying what fold each observation is in. If supplied, \code{n_folds} can
#' be missing.
#' @param min_ratio Ratio between smallest and largest value of lambda. Default is 1e-2.
#' @param lmda_path_size Number of values for \code{lambda}, if generated automatically.
#' Default is 100.
#' @param offsets Offsets, default is \code{NULL}. If present, this is
#'     a fixed vector or matrix corresponding to the shape of the natural
#'     parameter, and is added to the fit.
#' @param progress_bar Progress bar. Default is \code{FALSE}.
#' @return an object of class \code{"cv.grpnet"} is returned, which is a list
#' with the ingredients of the cross-validation fit.
#' \item{lambda}{the values of \code{lambda} used in the
#' fits.}
#' \item{cvm}{The mean cross-validated deviance - a vector of length \code{length(lambda)}.}
#' \item{cvsd}{estimate of standard error of \code{cvm}.}
#' \item{cvup}{upper curve = \code{cvm+cvsd}.}
#' \item{cvlo}{lower curve = \code{cvm-cvsd}.}
#' \item{nzero}{number of non-zero coefficients at each \code{lambda}.}
#' \item{name}{a text string indicating type of measure (for plotting purposes).
#' Currently this is \code{"deviance"}}
#' \item{grpnet.fit}{a fitted grpnet object for the
#' full data.}
#' \item{lambda.min}{value of \code{lambda} that gives minimum \code{cvm}.}
#' \item{lambda.1se}{largest value of \code{lambda} such that
#' mean deviance is within 1 standard error of the minimum.}
#' \item{index}{a one column matrix with the indices of \code{lambda.min} and \code{lambda.1se} in the sequence of coefficients, fits etc.}
#'
#'
#' @author James Yang, Trevor Hastie, and  Balasubramanian Narasimhan \cr Maintainer: Trevor Hastie
#' \email{hastie@stanford.edu}
#'
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.\cr
#' Friedman, J., Hastie, T. and Tibshirani, R. (2008)
#' \emph{Regularization Paths for Generalized Linear Models via Coordinate
#' Descent (2010), Journal of Statistical Software, Vol. 33(1), 1-22},
#' \doi{10.18637/jss.v033.i01}.\cr
#' Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2011)
#' \emph{Regularization Paths for Cox's Proportional
#' Hazards Model via Coordinate Descent, Journal of Statistical Software, Vol.
#' 39(5), 1-13},
#' \doi{10.18637/jss.v039.i05}.\cr
#' Tibshirani,Robert, Bien, J., Friedman, J., Hastie, T.,Simon, N.,Taylor, J. and
#' Tibshirani, Ryan. (2012) \emph{Strong Rules for Discarding Predictors in
#' Lasso-type Problems, JRSSB, Vol. 74(2), 245-266},
#' \url{https://arxiv.org/abs/1011.2234}.\cr

#' @examples
#' set.seed(0)
#' n <- 100
#' p <- 200
#' X <- matrix(rnorm(n * p), n, p)
#' y <- X[,1] * rnorm(1) + rnorm(n)
#' fit <- grpnet(X, glm.gaussian(y))
#' print(fit)
#'
#' @export cv.grpnet

cv.grpnet = function(
                     X,
                     glm,
                     n_folds = 10,
                     foldid = NULL,
                     min_ratio=1e-2,
                     lmda_path_size = 100,
                     offsets=NULL,
                     progress_bar= FALSE,
                    ...
                    ){
    cv.call = match.call()
    if(inherits(X,"sparseMatrix")){
        X <- as(X,"CsparseMatrix")
        X <- matrix.sparse(X, method="naive", n_threads=n_threads)
    }
    if (is.matrix(X) || is.array(X) || is.data.frame(X)) {
        X <- matrix.dense(X, method="naive", n_threads=n_threads)
    }

    N <- X$rows
    if (is.null(foldid))
        foldid = sample(rep(seq(n_folds), length = N))
  else n_folds = max(foldid)
  if (n_folds < 3)
    stop("n_folds must be bigger than 3; n_folds=10 recommended")
### get full-model lambda sequence
    fit_full = grpnet(X,glm,
                      min_ratio=min_ratio,
                      lmda_path_size = 100,
                      offsets=offsets,
                      progress_bar=progress_bar,
                      ...)
    lambda = fit_full$state$lmdas
    weights = glm$weights
### Set up deviance matrix
    devmat = matrix(NA,n_folds,length(lambda))
### Now loop over the folds
    logstep = seq(from = 0,to = log(min_ratio),length=lmda_path_size)
    for(k in 1:n_folds){
        local_weights=weights
        local_weights[foldid==k]=0
        local_glm = reweight(glm,local_weights)
        ## get local starting lambda
        fit0 = grpnet(X,local_glm,
                      lmda_path_size=1,
                      offsets=offsets,
                      progress_bar=FALSE,...)
        lam0 = fit0$state$lmdas
        local_lam = exp(log(lam0)-logstep)
        local_lam = local_lam[local_lam < lambda]
        aug_lam = c(local_lam, lambda)
        ## Now compute the fit
        fit_local = grpnet(X,local_glm,
                           lambda=aug_lam,
                           offsets=offsets,
                           progress_bar=progress_bar,
                           ...)
        pred_local = predict(fit_local,newx=X, lambda=lambda,newoffsets=offsets)
        devmat[k,] = getdev(pred_local,foldid==k,glm,local_glm)
    }
    fweights = tapply(weights,foldid,sum)
    cvm = apply(devmat,2,weighted.mean,w=fweights)
    cvsd = sqrt(apply(scale(devmat, cvm, FALSE)^2,
                      2, weighted.mean, w = fweights, na.rm = TRUE)/(n_folds - 1))
    nz = coef(fit_full)$df
    out = list(lambda = lambda, cvm = cvm, cvsd = cvsd,
               cvup = cvm + cvsd, cvlo = cvm - cvsd, nzero = nz)
    out = c(out,list(call = cv.call, name = "Mean Deviance", grpnet.fit = fit_full))
    lamin = with(out,getOptcv.grpnet(lambda, cvm, cvsd))
    out = c(out, as.list(lamin))
    class(out) = "cv.grpnet"
    out
    }




    getOptcv.grpnet = function(lambda, cvm, cvsd)
    {
        cvmin = min(cvm, na.rm = TRUE)
        idmin = cvm <= cvmin
        lambda.min = max(lambda[idmin], na.rm = TRUE)
        idmin = match(lambda.min, lambda)
        semin = (cvm + cvsd)[idmin]
        id1se = cvm <= semin
        lambda.1se = max(lambda[id1se], na.rm = TRUE)
        id1se = match(lambda.1se, lambda)
        index = matrix(c(idmin, id1se), 2, 1, dimnames = list(c("min",
        "1se"), "Lambda"))
        list(lambda.min = lambda.min, lambda.1se = lambda.1se, index = index)
    }

        getdev = function(eta,outfold,glm,local_glm){
            ## Helper function in cv.grpnet
            ## outfold is logical, with TRUE if in the leftout fold
            weights=glm$weights
            lossf = c(glm$loss_full(),local_glm$loss_full())
            if(glm$is_multi){
                ## eta is a n x k x lmda_path_size array
                dev = apply(eta,3,function(x)c(glm$loss(t(x)),local_glm$loss(t(x))))
            }
            else {
                ## eta is n x lmda_path_size array
                dev = apply(eta,2,function(x)c(glm$loss(x),local_glm$loss(x)))
            }
            dev = (dev[1,]-lossf[1]) - (dev[2,]-lossf[2])*sum(weights[!outfold])
            weights_sum_val = sum(weights[outfold])
            if(weights_sum_val>0) 2*dev/weights_sum_val else 0
            }





#' plot the cross-validation curve produced by cv.grpnet
#'
#' Plots the cross-validation curve, and upper and lower standard deviation
#' curves, as a function of the \code{lambda} values used.
#'
#' A plot is produced, and nothing is returned.
#'
#' @aliases plot.cv.grpnet
#' @param x fitted \code{"cv.grpnet"} object
#' @param sign.lambda Either plot against \code{log(lambda)} or its
#' negative (default) if \code{sign.lambda=-1}
#' @param \dots Other graphical parameters
#' @author Trevor Hastie and James Yang\cr Maintainer:
#' Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{grpnet} and \code{cv.grpnet}.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.\cr
#' Adelie Python user guide  \url{https://jamesyang007.github.io/adelie/}
#' @keywords models regression group lasso
#' @examples
#'
#' set.seed(1010)
#' n = 1000
#' p = 100
#' nzc = trunc(p/10)
#' x = matrix(rnorm(n * p), n, p)
#' beta = rnorm(nzc)
#' fx = (x[, seq(nzc)] %*% beta)
#' eps = rnorm(n) * 5
#' y = drop(fx + eps)
#' px = exp(fx)
#' px = px/(1 + px)
#' ly = rbinom(n = length(px), prob = px, size = 1)
#' cvob1 = cv.grpnet(x, glm.gaussian(y))
#' plot(cvob1)
#' title("Gaussian Family", line = 2.5)
#' frame()
#' set.seed(1011)
#' cvob2 = cv.grpnet(x, glm.binomial(ly))
#' plot(cvob2)
#' title("Binomial Family", line = 2.5)
#'
#' @method plot cv.grpnet
#' @export
plot.cv.grpnet=function(x,sign.lambda=-1,...){
    cvobj=x
    xlab = if(sign.lambda<0)
               expression(-Log(lambda))
           else
               expression(Log(lambda))

    fam = x$grpnet.fit$family
    name = paste0(cvobj$name," (",stringr::str_to_title(fam),")")
    plot.args=list(x=sign.lambda*log(cvobj$lambda),
                   y=cvobj$cvm,
                   ylim=range(cvobj$cvup,cvobj$cvlo),
                   xlab=xlab,
                   ylab=name,type="n")
  new.args=list(...)
  if(length(new.args))plot.args[names(new.args)]=new.args
do.call("plot",plot.args)
    error.bars(sign.lambda*log(cvobj$lambda),cvobj$cvup,cvobj$cvlo,width=0.01,
               col="darkgrey")
    points(sign.lambda*log(cvobj$lambda),cvobj$cvm,pch=20,
          col="red")
axis(side=3,at=sign.lambda*log(cvobj$lambda),labels=paste(cvobj$nz),tick=FALSE,line=0)
abline(v=sign.lambda*log(cvobj$lambda.min),lty=3)
abline(v=sign.lambda*log(cvobj$lambda.1se),lty=3)
  invisible()
}


#' plot coefficients from a "grpnet" object
#'
#' Produces a coefficient profile plot of the coefficient paths for a fitted
#' \code{"grpnet"} object.
#'
#' A coefficient profile plot is produced. If \code{x} is a multinomial or multigaussian model,
#' the 2norm of the vector of coefficients is plotted.
#'
#' @param x fitted \code{"grpnet"} model
#' @param sign.lambda This determines whether we plot against \code{log(lambda)} or its negative.
#' values are \code{-1}(default) or \code{1}
#' @param glm.name This is a logical (default \code{TRUE}), and causes the glm name of the model
#' to be included in the plot.
#' @param \dots Other graphical parameters to plot
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{grpnet}, and \code{print}, and \code{coef} methods, and
#' \code{cv.grpnet}.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.
#' @keywords models regression
#'
#' @examples
#' x=matrix(rnorm(100*20),100,20)
#' y=rnorm(100)
#' fit1=grpnet(x,glm.gaussian(y))
#' plot(fit1)
#' g4=diag(4)[sample(1:4,100,replace=TRUE),]
#' fit2=grpnet(x,glm.multinomial(g4))
#' plot(fit2,lwd=3)
#' fit3=grpnet(x,glm.gaussian(y),groups=c(1,5,9,13,17))
#' plot(fit3)

#' @method plot grpnet
#' @export
#' @export plot.grpnet
#' @importFrom stringr str_to_title

plot.grpnet=function(x, sign.lambda=-1,glm.name=TRUE,...){
    betaob=coef(x)
    ## Check if multi
    pK=dim(betaob$intercepts)
    K=pK[2]
    nlam=pK[1]
    betas = as.matrix(betaob$betas)
    if(K>1){
        ## change to 2norms
        p = ncol(betas)/K
        betas = array(t(betas),c(K,p,nlam))
        betas = apply(betas,c(2,3),function(x)sqrt(sum(x^2)))
        betas = t(betas)
        ylab= "Coefficients (2norm)"
    }
    else ylab="Coefficients"
    which=nonzeroCoef(t(betas))
  nwhich=length(which)
  switch(nwhich+1,#we add one to make switch work
         "0"={
           warning("No plot produced since all coefficients zero")
           return()
         },
         "1"=warning("1 or less nonzero coefficients; grpnet plot is not meaningful")
         )
    betas=betas[,which,drop=FALSE]
    xlab = if(sign.lambda<0)
               expression(-Log(lambda))
           else
               expression(Log(lambda))
    df = betaob$df
    ## all coefficients in a group get the same color
    colors=x$group_sizes
    colors=rep(seq(length(colors)),colors)
    index=sign.lambda*log(betaob$lambda)

    matplot(index,
            betas,
            xlab=xlab,
            ylab=ylab,
            col=colors,
            type="l",lty=1,...)
    atdf=pretty(index)
    prettydf=approx(x=index,y=df,xout=atdf,rule=2,method="constant",f=0)$y
    axis(3,at=atdf,labels=prettydf,tcl=NA,cex.axis=.8)
    if(glm.name){
        legendat = switch(as.character(sign.lambda),
                          "-1" = "topleft",
                          "1" = "topright"
                          )
        legendtext= stringr::str_to_title(x$family)
        legend(legendat, legend=legendtext,bty="n")
        }
    invisible()
    }


#' make predictions from a "cv.grpnet" object.
#'
#' This function makes predictions from a cross-validated grpnet model, using
#' the stored \code{"grpnet.fit"} object, and the optimal value chosen for
#' \code{lambda}.
#'
#' This function makes it easier to use the results of cross-validation to make
#' a prediction.
#'
#' @aliases coef.cv.grpnet predict.cv.grpnet
#' @param object Fitted \code{"cv.grpnet"}.
#' @param newx Matrix of new values for \code{x} at which predictions are to be
#' made. Can be a matrix, a sparse matrix as in \code{Matrix} package,
#' or else any of the matrix forms allowable in the \code{adelie} package. This
#' argument is not used for \code{type="coefficients"}.
#' @param lambda Value(s) of the penalty parameter \code{lambda} at which
#' predictions are required. Default is the value \code{lambda="lambda.1se"} stored
#' on the CV \code{object}. Alternatively \code{lambda="lambda.min"} can be used. If
#' \code{lambda} is numeric, it is taken as the value(s) of \code{lambda} to be
#' used.
#' @param \dots Not used. Other arguments to predict.
#' @return The object returned depends on the arguments.
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{grpnet}, and \code{print}, and \code{coef} methods, and
#' \code{cv.grpnet}.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.
#' @keywords models regression
#' @examples
#'
#' x = matrix(rnorm(100 * 20), 100, 20)
#' y = rnorm(100)
#' cv.fit = cv.grpnet(x, glm.gaussian(y))
#' predict(cv.fit, newx = x[1:5, ])
#' coef(cv.fit)
#' coef(cv.fit, lambda = "lambda.min")
#' predict(cv.fit, newx = x[1:5, ], lambda = c(0.001, 0.002))
#'
#' @method predict cv.grpnet
#' @export
predict.cv.grpnet=function(object,newx,lambda=c("lambda.1se","lambda.min"),...){
    if(is.character(lambda)){
        lambda=match.arg(lambda)
        namel = lambda
        lambda=object[[lambda]]
        names(lambda)=namel
    }
    predict(object$grpnet.fit,newx,lambda=lambda,...)
}

#' print a cross-validated grpnet object
#'
#' Print a summary of the results of cross-validation for a grpnet model.
#'
#' @param x fitted 'cv.grpnet' object
#' @param digits significant digits in printout
#' @param \dots additional print arguments
#' @author Trevor Hastie and James Yang\cr Maintainer: Trevor Hastie <hastie@@stanford.edu>
#' @seealso \code{grpnet}, \code{predict} and \code{coef} methods.
#' @references Yang, James and Hastie, Trevor. (2024) A Fast and Scalable Pathwise-Solver for Group Lasso
#' and Elastic Net Penalized Regression via Block-Coordinate Descent. arXiv \doi{10.48550/arXiv.2405.08631}.
#' @keywords models regression group lasso
#' @examples
#'
#' x = matrix(rnorm(100 * 20), 100, 20)
#' y = rnorm(100)
#' fit1 = cv.grpnet(x, glm.gaussian(y))
#' print(fit1)
#' @method print cv.grpnet
#' @export
#' @export print.cv.grpnet
print.cv.grpnet <- function(x, digits = max(3, getOption("digits") - 3), ...)
{
    cat("\nCall: ", deparse(x$call), "\n\n")

    optlams=c(x$lambda.min,x$lambda.1se)
    which=match(optlams,x$lambda)
    mat = with(x, cbind(optlams, which, cvm[which], cvsd[which], nzero[which]))
    dimnames(mat) = list(c("min", "1se"), c("Lambda", "Index","Measure",
                                            "SE", "Nonzero"))
    cat("Measure:", x$name,"\n\n")

    mat=data.frame(mat,check.names=FALSE)
    class(mat)=c("anova",class(mat))
    print(mat,digits=digits)
}

#' @method coef cv.grpnet
#' @export
coef.cv.grpnet=function(object,lambda=c("lambda.1se","lambda.min"),...){
    if(is.character(lambda)){
      lambda=match.arg(lambda)
      lambda=object[[lambda]]
    }
  coef(object$grpnet.fit,lambda=lambda,...)
}

### Utility function stolen from glmnet
nonzeroCoef = function (beta, bystep = FALSE)
{
### bystep = FALSE means which variables were ever nonzero
### bystep = TRUE means which variables are nonzero for each step
  nr=nrow(beta)
  if (nr == 1) {#degenerate case
    if (bystep)
      apply(beta, 2, function(x) if (abs(x) > 0)
            1
      else NULL)
    else {
      if (any(abs(beta) > 0))
        1
      else NULL
    }
  }
  else {
    beta=abs(beta)>0 # this is sparse
    which=seq(nr)
    ones=rep(1,ncol(beta))
    nz=as.vector((beta%*%ones)>0)
    which=which[nz]
    if (bystep) {
      if(length(which)>0){
        beta=as.matrix(beta[which,,drop=FALSE])
        nzel = function(x, which) if (any(x))
          which[x]
        else NULL
        which=apply(beta, 2, nzel, which)
        if(!is.list(which))which=data.frame(which)# apply can return a matrix!!
        which
      }
      else{
        dn=dimnames(beta)[[2]]
        which=vector("list",length(dn))
        names(which)=dn
        which
      }

    }
    else which
  }
}
