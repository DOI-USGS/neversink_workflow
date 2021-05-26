module SmoothingModule
  use KindModule, only: DP, I4B
  use ConstantsModule, only: DZERO, DHALF, DONE, DTWO, DTHREE, DFOUR,            &
 &                           DSIX, DPREC, DEM2, DEM4, DEM5, DEM6, DEM8, DEM14 
  implicit none
  
  contains
    
  subroutine sSCurve(x,range,dydx,y)
! ******************************************************************************
! COMPUTES THE S CURVE FOR SMOOTH DERIVATIVES BETWEEN X=0 AND X=1
! FROM mfusg smooth SUBROUTINE in gwf2wel7u1.f
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(in) :: x
    real(DP), intent(in) :: range
    real(DP), intent(inout) :: dydx
    real(DP), intent(inout) :: y
    !--local variables
    real(DP) :: s
    real(DP) :: xs
! ------------------------------------------------------------------------------
!   code
!
    s = range
    if ( s < DPREC ) s = DPREC
    xs = x / s
    if (xs < DZERO) xs = DZERO
    if (xs <= DZERO) then
      y = DZERO
      dydx = DZERO
    elseif(xs < DONE)then
      y = -DTWO * xs**DTHREE + DTHREE * xs**DTWO
      dydx = -DSIX * xs**DTWO + DSIX * xs
    else
      y = DONE
      dydx = DZERO
    endif
    return
  end subroutine sSCurve
  
  subroutine sCubicLinear(x,range,dydx,y)
! ******************************************************************************
! COMPUTES THE S CURVE WHERE DY/DX = 0 at X=0; AND DY/DX = 1 AT X=1.
! Smooths from zero to a slope of 1.
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(in) :: x
    real(DP), intent(in) :: range
    real(DP), intent(inout) :: dydx
    real(DP), intent(inout) :: y
    !--local variables
    real(DP) :: s
    real(DP) :: xs
! ------------------------------------------------------------------------------
!   code
!
    s = range
    if ( s < DPREC ) s = DPREC
    xs = x / s
    if (xs < DZERO) xs = DZERO
    if (xs <= DZERO) then
      y = DZERO
      dydx = DZERO
    elseif(xs < DONE)then
      y = -DONE * xs**DTHREE + DTWO * xs**DTWO
      dydx = -DTHREE * xs**DTWO + DFOUR * xs
    else
      y = DONE
      dydx = DZERO
    endif
    return
  end subroutine sCubicLinear

  subroutine sCubic(x,range,dydx,y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1; cubic function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(inout) :: x
    real(DP), intent(inout) :: range
    real(DP), intent(inout) :: dydx
    real(DP), intent(inout) :: y
    !--local variables
    real(DP) :: s, aa, bb
    real(DP) :: cof1, cof2, cof3
! ------------------------------------------------------------------------------
!   code
!
    dydx = DZERO
    y = DZERO
    if ( range < DPREC ) range = DPREC
    if ( x < DPREC ) x = DPREC
    s = range
    aa = -DSIX/(s**DTHREE)
    bb = -DSIX/(s**DTWO)
    cof1 = x**DTWO
    cof2 = -(DTWO*x)/(s**DTHREE)
    cof3 = DTHREE/(s**DTWO)
    y = cof1 * (cof2 + cof3)
    dydx = (aa*x**DTWO - bb*x)
    if ( x <= DZERO ) then
      y = DZERO
      dydx = DZERO
    else if ( (x - s) > -DPREC ) then
      y = DONE
      dydx = DZERO
    end if
    return
  end subroutine sCubic
  
  subroutine sLinear(x,range,dydx,y)
! ******************************************************************************
! Linear smoothing function returns value between 0-1
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(inout) :: x
    real(DP), intent(inout) :: range
    real(DP), intent(inout) :: dydx
    real(DP), intent(inout) :: y
    !--local variables
    real(DP) :: s
! ------------------------------------------------------------------------------
!   code
!
    dydx = DZERO
    y = DZERO
    if ( range < DPREC ) range = DPREC
    if ( x < DPREC ) x = DPREC
    s = range
    y = DONE - (s - x)/s
    dydx = DONE/s
    if ( y > DONE ) then
      y = DONE
      dydx = DZERO
    end if
    return
  end subroutine sLinear
    
  subroutine sQuadratic(x,range,dydx,y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1; quadratic function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(inout) :: x
    real(DP), intent(inout) :: range
    real(DP), intent(inout) :: dydx
    real(DP), intent(inout) :: y
    !--local variables
    real(DP) :: s
! ------------------------------------------------------------------------------
!   code
!
    dydx = DZERO
    y = DZERO
    if ( range < DPREC ) range = DPREC
    if ( x < DPREC ) x = DPREC
    s = range
    y = (x**DTWO) / (s**DTWO)
    dydx = DTWO*x/(s**DTWO)
    if ( y > DONE ) then
      y = DONE
      dydx = DZERO
    end if
    return
  end subroutine sQuadratic

  subroutine sChSmooth(d, smooth, dwdh)
! ******************************************************************************
! Function to smooth channel variables during channel drying
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    real(DP), intent(in) :: d
    real(DP), intent(inout) :: smooth
    real(DP), intent(inout) :: dwdh
    !
    ! -- local variables
    real(DP) :: s
    real(DP) :: diff
    real(DP) :: aa
    real(DP) :: ad
    real(DP) :: b
    real(DP) :: x
    real(DP) :: y
! ------------------------------------------------------------------------------
!   code
!    
    smooth = DZERO
    s = DEM5
    x = d
    diff = x - s
    if ( diff > DZERO ) then
      smooth = DONE
      dwdh = DZERO
    else
      aa = -DONE / (s**DTWO)
      ad = -DTWO / (s**DTWO)
      b = DTWO / s
      y = aa * x**DTWO + b*x
      dwdh = (ad*x + b)
      if ( x <= DZERO ) then
        y = DZERO
        dwdh = DZERO
      else if ( diff > -DEM14 ) then
        y = DONE
        dwdh = DZERO
      end if
      smooth = y
    end if
    return
end subroutine sChSmooth
 
  function sLinearSaturation(top, bot, x) result(y)
! ******************************************************************************
! Linear smoothing function returns value between 0-1;
! Linear saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    ! -- local
    real(DP) :: b
! ------------------------------------------------------------------------------
!   code
!
    b = top - bot
    if (x < bot) then
      y = DZERO
    else if (x > top) then
      y = DONE
    else
      y = (x - bot) / b
    end if
    return
  end function sLinearSaturation


  function sCubicSaturation(top, bot, x, eps) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1;
! Quadratic saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), intent(in), optional :: eps
    ! -- local
    real(DP) :: teps
    real(DP) :: w
    real(DP) :: b
    real(DP) :: s
    real(DP) :: cof1
    real(DP) :: cof2
! ------------------------------------------------------------------------------
!   code
!
    if (present(eps)) then
      teps = eps
    else
      teps = DEM2
    end if
    w = x - bot
    b = top - bot
    s = teps * b
    cof1 = DONE / (s**DTWO)
    cof2 = DTWO / s
    if (w < DZERO) then
      y = DZERO
    else if (w < s) then
      y = -cof1 * (w**DTHREE) + cof2 * (w**DTWO)
    else if (w < (b-s)) then
      y = w / b
    else if (w < b) then
      y = DONE + cof1 * ((b - w)**DTHREE) - cof2 * ((b - w)**DTWO)
    else
      y = DONE
    end if
    
    return
  end function sCubicSaturation

  
  function sQuadraticSaturation(top, bot, x, eps, bmin) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1;
! Quadratic saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), optional, intent(in) :: eps
    real(DP), optional, intent(in) :: bmin
    ! -- local
    real(DP) :: teps
    real(DP) :: tbmin
    real(DP) :: b
    real(DP) :: br
    real(DP) :: bri
    real(DP) :: av
! ------------------------------------------------------------------------------
!   code
!
    if (present(eps)) then
      teps = eps
    else
      teps = DEM6
    end if
    if (present(bmin)) then
      tbmin = bmin
    else
      tbmin = DZERO
    end if
    b = top - bot
    if (b > DZERO) then
      if (x < bot) then
        br = DZERO
      else if (x > top) then
        br = DONE
      else
        br = (x - bot) / b
      end if
      av = DONE / (DONE - teps) 
      bri = DONE - br
      if (br < tbmin) then
        br = tbmin
      end if
      if (br < teps) then
        y = av * DHALF * (br*br) / teps
      elseif (br < (DONE-teps)) then
        y = av * br + DHALF * (DONE - av)
      elseif (br < DONE) then
        y = DONE - ((av * DHALF * (bri * bri)) / teps)
      else
        y = DONE
      end if
    else
      if (x < bot) then
        y = DZERO
      else
        y = DONE
      end if
    end if
    
    return
  end function sQuadraticSaturation
  
  function svanGenuchtenSaturation(top, bot, x, alpha, beta, sr) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1;
! van Genuchten saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), intent(in) :: alpha
    real(DP), intent(in) :: beta
    real(DP), intent(in) :: sr
    ! -- local
    real(DP) :: b
    real(DP) :: pc
    real(DP) :: gamma
    real(DP) :: seff
! ------------------------------------------------------------------------------
!   code
!
    b = top - bot
    pc = (DHALF * b) - x
    if (pc <= DZERO) then
      y = DZERO
    else
      gamma = DONE - (DONE / beta)
      seff = (DONE + (alpha * pc)**beta)**gamma
      seff = DONE / seff
      y = seff * (DONE - sr) + sr
    end if

    return
  end function svanGenuchtenSaturation
 
  
  function sQuadraticSaturationDerivative(top, bot, x, eps, bmin) result(y)
! ******************************************************************************
! Derivative of nonlinear smoothing function returns value between 0-1;
! Derivative of the quadratic saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), optional, intent(in) :: eps
    real(DP), optional, intent(in) :: bmin
    ! -- local
    real(DP) :: teps
    real(DP) :: tbmin
    real(DP) :: b
    real(DP) :: br
    real(DP) :: bri
    real(DP) :: av
! ------------------------------------------------------------------------------
!   code
!
    if (present(eps)) then
      teps = eps
    else
      teps = DEM6
    end if
    if (present(bmin)) then
      tbmin = bmin
    else
      tbmin = DZERO
    end if
    b = top - bot
    if (x < bot) then
      br = DZERO
    else if (x > top) then
      br = DONE
    else
      br = (x - bot) / b
    end if
    av = DONE / (DONE - teps) 
    bri = DONE - br
    if (br < tbmin) then
      br = tbmin
    end if
    if (br < teps) then
      y = av * br / teps
    elseif (br < (DONE-teps)) then
      y = av
    elseif (br < DONE) then
      y = av * bri / teps
    else
      y = DZERO
    end if
    y = y / b
    
    return
  end function sQuadraticSaturationDerivative



  function sQSaturation(top, bot, x, c1, c2) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1;
! Cubic saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), intent(in), optional :: c1
    real(DP), intent(in), optional :: c2
    ! -- local
    real(DP) :: w
    real(DP) :: b
    real(DP) :: s
    real(DP) :: cof1
    real(DP) :: cof2
! ------------------------------------------------------------------------------
!   code
!
    !
    ! -- process optional variables
    if (present(c1)) then
      cof1 = c1
    else
      cof1 = -DTWO
    end if
    if (present(c2)) then
      cof2 = c2
    else
      cof2 = DTHREE
    end if
    !
    ! -- calculate head diference from bottom (w),
    !    calculate range (b), and
    !    calculate normalized head difference from bottom (s)
    w = x - bot
    b = top - bot
    s = w / b
    !
    ! -- divide cof1 and cof2 by range to the power 3 and 2, respectively
    cof1 = cof1 / b**DTHREE
    cof2 = cof2 / b**DTWO
    !
    ! -- calculate fraction
    if (s < DZERO) then
      y = DZERO
    else if(s < DONE) then
      y = cof1 * w**DTHREE + cof2 * w**DTWO
    else
      y = DONE
    end if
    !
    ! -- return
    return
  end function sQSaturation
  
  function sQSaturationDerivative(top, bot, x, c1, c2) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns value between 0-1;
! Cubic saturation function
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: top
    real(DP), intent(in) :: bot
    real(DP), intent(in) :: x
    real(DP), intent(in), optional :: c1
    real(DP), intent(in), optional :: c2
    ! -- local
    real(DP) :: w
    real(DP) :: b
    real(DP) :: s
    real(DP) :: cof1
    real(DP) :: cof2
! ------------------------------------------------------------------------------
!   code
!
    !
    ! -- process optional variables
    if (present(c1)) then
      cof1 = c1 
    else
      cof1 = -DTWO
    end if
    if (present(c2)) then
      cof2 = c2
    else
      cof2 = DTHREE
    end if
    !
    ! -- calculate head diference from bottom (w),
    !    calculate range (b), and
    !    calculate normalized head difference from bottom (s)
    w = x - bot
    b = top - bot
    s = w / b
    !
    ! -- multiply cof1 and cof2 by 3 and 2, respectively, and then 
    !    divide by range to the power 3 and 2, respectively
    cof1 = cof1 * DTHREE / b**DTHREE
    cof2 = cof2 * DTWO / b**DTWO
    !
    ! -- calculate derivative of fraction with respect to x
    if (s < DZERO) then
      y = DZERO
    else if(s < DONE) then
      y = cof1 * w**DTWO + cof2 * w
    else
      y = DZERO
    end if
    !
    ! -- return
    return
  end function sQSaturationDerivative
  
  function sSlope(x, xi, yi, sm, sp, ta) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns a smoothed value of y that has the value
! yi at xi and yi + (sm * dx) for x-values less than xi and yi + (sp * dx) for
! x-values greater than xi, where dx = x - xi.
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), intent(in) :: yi
    real(DP), intent(in) :: sm
    real(DP), intent(in) :: sp
    real(DP), optional, intent(in) :: ta
    ! -- local
    real(DP) :: a
    real(DP) :: b
    real(DP) :: dx
    real(DP) :: xm
    real(DP) :: xp
    real(DP) :: ym
    real(DP) :: yp
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing variable a
    if (present(ta)) then
      a = a
    else
      a = DEM8
    end if
    !
    ! -- calculate b from smoothing variable a
    b = a / (sqrt(DTWO) - DONE)
    !
    ! -- calculate contributions to y
    dx = x - xi
    xm = DHALF * (x + xi - sqrt(dx + b**DTWO - a**DTWO))
    xp = DHALF * (x + xi + sqrt(dx + b**DTWO - a**DTWO))
    ym = sm * (xm - xi)
    yp = sp * (xi - xp)
    !
    ! -- calculate y from ym and yp contributions
    y = yi + ym + yp
    !
    ! -- return
    return
  end function sSlope  
    
  function sSlopeDerivative(x, xi, sm, sp, ta) result(y)
! ******************************************************************************
! Derivative of nonlinear smoothing function that has the value yi at xi and 
! yi + (sm * dx) for x-values less than xi and yi + (sp * dx) for x-values 
! greater than xi, where dx = x - xi.
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), intent(in) :: sm
    real(DP), intent(in) :: sp
    real(DP), optional, intent(in) :: ta
    ! -- local
    real(DP) :: a
    real(DP) :: b
    real(DP) :: dx
    real(DP) :: mu
    real(DP) :: rho
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing variable a
    if (present(ta)) then
      a = a
    else
      a = DEM8
    end if
    !
    ! -- calculate b from smoothing variable a
    b = a / (sqrt(DTWO) - DONE)
    !
    ! -- calculate contributions to derivative
    dx = x - xi
    mu = sqrt(dx**DTWO + b**DTWO - a**DTWO)
    rho = dx / mu
    !
    ! -- calculate derivative from individual contributions
    y = DHALF * (sm + sp) - DHALF * rho * (sm - sp)                     
    !
    ! -- return
    return
  end function sSlopeDerivative  
  
  function sQuadratic0sp(x, xi, tomega) result(y)
! ******************************************************************************
! Nonlinear smoothing function returns a smoothed value of y that uses a 
! quadratic to smooth x over range of xi - epsilon to xi + epsilon.
! Simplification of sQuadraticSlope with sm = 0, sp = 1, and yi = 0.
! From Panday et al. (2013) - eq. 35 - https://dx.doi.org/10.5066/F7R20ZFJ
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), optional, intent(in) :: tomega
    ! -- local
    real(DP) :: omega
    real(DP) :: epsilon
    real(DP) :: dx
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing interval
    if (present(tomega)) then
      omega = tomega
    else
      omega = DEM6
    end if
    !
    ! -- set smoothing interval
    epsilon = DHALF * omega
    !
    ! -- calculate distance from xi
    dx = x - xi
    !
    ! -- evaluate smoothing function
    if (dx < -epsilon) then
      y = xi
    else if (dx < epsilon) then
      y = (dx**DTWO / (DFOUR * epsilon)) + DHALF * dx + (epsilon / DFOUR) + xi
    else
      y = x
    end if
    !
    ! -- return
    return
  end function sQuadratic0sp  
  
  function sQuadratic0spDerivative(x, xi, tomega) result(y)
! ******************************************************************************
! Derivative of nonlinear smoothing function returns a smoothed value of y  
! that uses a quadratic to smooth x over range of xi - epsilon to xi + epsilon.
! Simplification of sQuadraticSlope with sm = 0, sp = 1, and yi = 0.
! From Panday et al. (2013) - eq. 35 - https://dx.doi.org/10.5066/F7R20ZFJ
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), optional, intent(in) :: tomega
    ! -- local
    real(DP) :: omega
    real(DP) :: epsilon
    real(DP) :: dx
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing interval
    if (present(tomega)) then
      omega = tomega
    else
      omega = DEM6
    end if
    !
    ! -- set smoothing interval
    epsilon = DHALF * omega
    !
    ! -- calculate distance from xi
    dx = x - xi
    !
    ! -- evaluate smoothing function
    if (dx < -epsilon) then
      y = 0
    else if (dx < epsilon) then
      y = (dx / omega) + DHALF
    else
      y = 1
    end if
    !
    ! -- return
    return
  end function sQuadratic0spDerivative  
  
  function sQuadraticSlope(x, xi, yi, sm, sp, tomega) result(y)
! ******************************************************************************
! Quadratic smoothing function returns a smoothed value of y that has the value
! yi at xi and yi + (sm * dx) for x-values less than xi and yi + (sp * dx) for
! x-values greater than xi, where dx = x - xi.
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), intent(in) :: yi
    real(DP), intent(in) :: sm
    real(DP), intent(in) :: sp
    real(DP), optional, intent(in) :: tomega
    ! -- local
    real(DP) :: omega
    real(DP) :: epsilon
    real(DP) :: dx
    real(DP) :: c
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing interval
    if (present(tomega)) then
      omega = tomega
    else
      omega = DEM6
    end if
    !
    ! -- set smoothing interval
    epsilon = DHALF * omega
    !
    ! -- calculate distance from xi
    dx = x - xi
    !
    ! -- evaluate smoothing function
    if (dx < -epsilon) then
      y = sm * dx
    else if (dx < epsilon) then
      c = dx / epsilon
      y = DHALF * epsilon * (DHALF * (sp - sm) * (DONE + c**DTWO) + (sm + sp) * c)
    else
      y = sp * dx
    end if
    !
    ! -- add value at xi
    y = y + yi
    !
    ! -- return
    return
  end function sQuadraticSlope  
  
  
  function sQuadraticSlopeDerivative(x, xi, sm, sp, tomega) result(y)
! ******************************************************************************
! Derivative of quadratic smoothing function returns a smoothed value of y 
! that has the value yi at xi and yi + (sm * dx) for x-values less than xi and 
! yi + (sp * dx) for x-values greater than xi, where dx = x - xi.
! ******************************************************************************
! 
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- return
    real(DP) :: y
    ! -- dummy variables
    real(DP), intent(in) :: x
    real(DP), intent(in) :: xi
    real(DP), intent(in) :: sm
    real(DP), intent(in) :: sp
    real(DP), optional, intent(in) :: tomega
    ! -- local
    real(DP) :: omega
    real(DP) :: epsilon
    real(DP) :: dx
    real(DP) :: c
! ------------------------------------------------------------------------------
    !
    ! -- set smoothing interval
    if (present(tomega)) then
      omega = tomega
    else
      omega = DEM6
    end if
    !
    ! -- set smoothing interval
    epsilon = DHALF * omega
    !
    ! -- calculate distance from xi
    dx = x - xi
    !
    ! -- evaluate smoothing function
    if (dx < -epsilon) then
      y = sm
    else if (dx < epsilon) then
      c = dx / epsilon
      y = DHALF * ((sp - sm) * c + (sm + sp))
    else
      y = sp
    end if
    !
    ! -- return
    return
  end function sQuadraticSlopeDerivative 
  
end module SmoothingModule
    
    