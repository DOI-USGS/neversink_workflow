!========================================================================================
!==== This code is the main driver for conducting basic Particle Swarm Optimization  ====
!==== for parameter estimation within the PEST suite, using the YAMR run manager.    ====
!====    by Adam Siade                                                               ====
!========================================================================================
!========================================================================================
program particleswarmopt

  use psodat

  implicit none

! specifications:
!----------------------------------------------------------------------------------------
!---------------------------------------------------------------------------------------- 
! external routines for run management via YAMR
!----------------------------------------------------------------------------------------
  external rmif_delete
  integer rmif_delete
  
!----------------------------------------------------------------------------------------
! local PSO-main variables
!----------------------------------------------------------------------------------------
  integer::err
  
  character(len=100)::pstnam,basnam,recnam
  character(len=20)::port
!----------------------------------------------------------------------------------------
!----------------------------------------------------------------------------------------

  err = 0
  
! read arguments from the command line
  call getarg(1,pstnam)
  call getarg(2,port)
  if (pstnam(1:1) == ' ' .or. port(1:1) == ' ') then
    write(*,*)'Usage is: psopp case.pst port'
    stop
  end if
  !
! record PEST base name
  basnam = pstnam
  basnam = adjustr(basnam)
  !
  if (basnam(97:100) /= '.pst') then
    write(*,*)'Error: PEST control file name must end in .pst'
    write(*,*)'-- stopping execution --'
    stop
  end if
  !
  basnam(97:100) = '    '
  basnam = adjustl(basnam)
  !
! read pest control file
  call readpst(pstnam)
  !
! read pso control file
  call readpso(pstnam)
  
  
! assign unit numbers
! -------------------------------------------------------------------------------------------------
  allocate(unit(10))
  unit = 0
  !
! record file
  unit(1) = 19
! restart output
  unit(6) = 24
! restart input
  unit(7) = 26
  !
! estimation mode
! ~~~~~~~~~~~~~~~
! gbest and pbest output
  unit(2) = 20
  unit(3) = 21
! observation output
  unit(4) = 22
  !
! parunc mode
! ~~~~~~~~~~~
  unit(9) = 28
  ujac    = 29 
  !
! pareto mode
! ~~~~~~~~~~~
! all pareto-specific output files
  unit(5) = 23
  !
! initial parameter sets
  uipar   = 27
!--------------------------------------------------------------------------------------------------
  

! open record file
  recnam = trim(basnam) // '.rec'
  open(unit(1),file=trim(recnam))
  !
! set up PEST++ run manager
! instantiate
  call instantrm(port)
  !                       
! initialize
  call initialrm(0)
  !
! if-statements governing pestmode
  if (trim(pestmode) == 'estimation') then
    !
    call psoest(basnam)
    !
  else if (trim(pestmode) == 'pareto') then
    !
    call psopareto(basnam)
    !
  else if (trim(pestmode) == 'parunc') then
    !
    call parunc(basnam)
    !
  end if
  
  
! clean up
  close(unit(1))
  !
  err = rmif_delete()
  !
  stop
  
  
end program particleswarmopt





















