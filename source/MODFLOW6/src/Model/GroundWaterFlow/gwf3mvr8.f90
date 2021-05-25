!GWF Water Mover Module
!This module contains a derived type, called GwfMvrType, that
!is attached to the GWF model.  The water mover can be used to move water
!between packages.  The mover requires that mover-aware packages have access
!to four arrays: qtformvr, qformvr, qtomvr, and qfrommvr.  These arrays are 
!stored and managed by a separate PackageMoverType object.  qformvr is a
!vector of volumetric flow rates available for the mover.  The package
!must fill the vector (dimensioned by number of reaches) with the available
!water.  qtomvr is a vector containing how much water was actually moved
!by the mover.  The package should use this value in the budgeting part
!to track how much water was actually provided to the mover.  Lastly,
!the qfrommvr is a vector that contains volumetric rates for how much
!water was provided by the mover as a source of water to the package.
!
!The mover is designed so that a reach can provide water to more than one
!receiving reaches.  The available water will be consumed in order of
!the movers listed in the package.  The mover is also designed so that
!a receiver can receive water from more than one provider.
!
!  1.  The mover is instantiated as a model member:
!
!      type(GwfMvrType),               pointer :: mvr     => null()
!
!      Mover aware packages have access to the following vectors of mover
!      information, which are stored in the PackageMoverType object:
!
!      qtformvr(nproviders) -- total available unconsumed water for mover
!      qformvr(nproviders) -- currently available consumed water (changes during fc)
!      qtomvr(nproviders) -- actual amount of water sent to mover
!      qfrommvr(nreceivers) -- actual amount of water received from mover
!
!      integer(I4B), pointer                        :: imover        => null()
!      real(DP), dimension(:), pointer, contiguous  :: qtformvr      => null()
!      real(DP), dimension(:), pointer, contiguous  :: qformvr       => null()
!      real(DP), dimension(:), pointer, contiguous  :: qtomvr        => null()
!      real(DP), dimension(:), pointer, contiguous  :: qfrommvr      => null()
!
!      Note qtformvr is filled as a positive number to indicate that it is
!      water available to be moved.  If qtformvr is negative, then
!      no water will be moved for that reach.  qformvr is also the available
!      water, but this value decreases as the mover object consumes water from
!      it.
!
!  2.  In gwf_cr create the mover package by calling the CR subroutine:
!
!      call mvr_cr(this%mvr, this%name, this%inmvr, this%iout)
!
!  3.  In gwf_ar call the AR method for the mover:
!
!      if(this%inmvr > 0) call this%mvr%mvr_ar()
!
!      Mover aware packages allocate the four vectors.  The first three 
!      (qtformvr, qformvr, qtomvr) are allocated to the number of providers
!      and the last one (qfrommvr) is allocated to the number of receivers.
!
!  4.  In gwf_rp call the RP method for the mover.  This reads the 
!      movers active for the current period.
!
!      if(this%inmvr > 0) call this%mvr%mvr_rp()
!
!  5.  In gwf_ad call the AD method for the mover.  This saves qtomvr from the
!      the last time step.
!
!      if(this%inmvr > 0) call this%mvr%mvr_ad()
!
!      Mover aware packages then set:
!        qtomvr(:) = 0.
!        qformvr(:) = 0.
!
!  6.  In gwf_cf call the CF routine. Mover aware packages set:
!        qtformvr(:) = qformvr(:)
!        qfrommvr(:) = 0.
!        qtomvr(:) = 0.
!
!  7.  The FC method for the mover is called.  This method calculates the
!      amount of water to move based on the amount of water available from the
!      previous iteration.  This call updates the values in the qtomvr and
!      qfrommvr vectors inside the packages.  This is done by the mover package
!      using pointers to the appropriate reach locations in qtomvr and qfrommvr.
!
!      if(this%inmvr > 0) call this%mvr%mvr_fc()  ! called from gwf%gwf_fc()
!
!      a. Mover aware packages first set qformvr(:) = 0.
!      b. Mover aware packages that are receivers (MAW, SFR, LAK, UZF) add 
!         qfrommvr terms to their individual control volume equations as a 
!         source of water.
!      c. Mover aware packages calculate qformvr as amount of water available
!         to be moved (these qformvr terms are used in the next iteration
!         by this%mvr%mvr_fc() to calculate how much water is actually moved)
!
!  8.  The BD method for the mover is called.  This method writes the moved
!      water rates if requested.
!
!      if(this%inmvr > 0) call this%mvr%mvr_bd()
!
!      Mover aware packages account for qtomvr and qfrommvr terms in their
!      individual budget routines.
!
!  9.  The OT method for the mover is called.  This method outputs a mover
!      budget table.
!
!      if(this%inmvr > 0) call this%mvr%mvr_ot()
!
module GwfMvrModule
  use KindModule,             only: DP, I4B
  use ConstantsModule,        only: LENMEMPATH, LENPACKAGENAME, LENMODELNAME,   &
                                    LENBUDTXT, LENAUXNAME, LENPAKLOC,          &
                                    DZERO, DNODATA, MAXCHARLEN, TABCENTER,     &
                                    LINELENGTH
  use MvrModule,              only: MvrType
  use BudgetModule,           only: BudgetType, budget_cr
  use BudgetObjectModule,     only: BudgetObjectType, budgetobject_cr
  use NumericalPackageModule, only: NumericalPackageType
  use BlockParserModule,      only: BlockParserType
  use PackageMoverModule,     only: PackageMoverType
  use BaseDisModule,          only: DisBaseType
  use InputOutputModule,      only: urword
  use TableModule,            only: TableType, table_cr

  implicit none
  private
  public :: GwfMvrType, mvr_cr

  type, extends(NumericalPackageType) :: GwfMvrType
    integer(I4B), pointer                            :: ibudgetout => null()     !< binary budget output file
    integer(I4B), pointer                            :: maxmvr => null()         !< max number of movers to be specified
    integer(I4B), pointer                            :: maxpackages => null()    !< max number of packages to be specified
    integer(I4B), pointer                            :: maxcomb => null()        !< max number of combination of packages
    integer(I4B), pointer                            :: nmvr => null()           !< number of movers for current stress period
    integer(I4B), pointer                            :: iexgmvr => null()        !< indicate mover is for an exchange (not for a single model)
    integer(I4B), pointer                            :: imodelnames => null()    !< indicate package input file has model names in it
    integer(I4B), dimension(:), pointer, contiguous  :: ientries => null()       !< number of entries for each combination
    character(len=LENMEMPATH),                                                &
      dimension(:), pointer, contiguous              :: pckMemPaths              !< memory paths of all packages used in this mover
    character(len=LENPACKAGENAME),                                             &
      dimension(:), pointer, contiguous              :: paknames => null()       !< array of package names
    type(MvrType), dimension(:), pointer, contiguous :: mvr => null()            !< array of movers
    type(BudgetType), pointer                        :: budget => null()         !< mover budget object (used to write table)
    type(BudgetObjectType), pointer                  :: budobj => null()         !< new budget container (used to write binary file)
    type(PackageMoverType),                                                    &
      dimension(:), pointer, contiguous    :: pakmovers => null()                !< pointer to package mover objects
    !
    ! -- table objects
    type(TableType), pointer :: outputtab => null()
  contains
    procedure :: mvr_ar
    procedure :: mvr_rp
    procedure :: mvr_ad
    procedure :: mvr_fc
    procedure :: mvr_cc
    procedure :: mvr_bd
    procedure :: mvr_ot
    procedure :: mvr_da
    procedure :: read_options
    procedure :: check_options
    procedure :: read_dimensions
    procedure :: read_packages
    procedure :: check_packages
    procedure :: assign_packagemovers
    procedure :: allocate_scalars
    procedure :: allocate_arrays
    procedure, private :: mvr_setup_budobj
    procedure, private :: mvr_fill_budobj
    procedure, private :: mvr_setup_outputtab
    procedure, private :: mvr_print_outputtab
  end type GwfMvrType

  contains

  subroutine mvr_cr(mvrobj, name_parent, inunit, iout, dis, iexgmvr)
! ******************************************************************************
! mvr_cr -- Create a new mvr object
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    type(GwfMvrType), pointer :: mvrobj
    character(len=*), intent(in) :: name_parent
    integer(I4B), intent(in) :: inunit
    integer(I4B), intent(in) :: iout
    class(DisBaseType), pointer, intent(in) :: dis
    integer(I4B), optional :: iexgmvr
! ------------------------------------------------------------------------------
    !
    ! -- Create the object
    allocate(mvrobj)
    !
    ! -- create name and memory paths. name_parent will either be model name or the
    !    exchange name.
    call mvrobj%set_names(1, name_parent, 'MVR', 'MVR')
    !
    ! -- Allocate scalars
    call mvrobj%allocate_scalars()
    !
    ! -- Set pointer to dis
    mvrobj%dis => dis
    !
    ! -- Set variables
    mvrobj%inunit = inunit
    mvrobj%iout = iout
    !
    ! -- Set iexgmvr
    if(present(iexgmvr)) mvrobj%iexgmvr = iexgmvr
    !
    ! -- Create the budget object
    if (inunit > 0) then
      call budget_cr(mvrobj%budget, mvrobj%memoryPath)
      !
      ! -- Initialize block parser
      call mvrobj%parser%Initialize(mvrobj%inunit, mvrobj%iout)
    endif
    !
    ! -- Return
    return
  end subroutine mvr_cr

  subroutine mvr_ar(this)
! ******************************************************************************
! mvr_ar -- Allocate and read water mover information
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- locals
! ------------------------------------------------------------------------------
    !
    ! -- Print a message identifying the water mover package.
    write(this%iout, 1) this%inunit
  1 format(1x,/1x,'MVR -- WATER MOVER PACKAGE, VERSION 8, 1/29/2016', &
      ' INPUT READ FROM UNIT ', i0)
    !
    ! -- Read and check options
    call this%read_options()
    call this%check_options()
    !
    ! -- Read options
    call this%read_dimensions()
    !
    ! -- Allocate arrays
    call this%allocate_arrays()
    !
    ! -- Read and check package names
    call this%read_packages()
    call this%check_packages()
    !
    ! -- Define the budget object to be the size of package names
    call this%budget%budget_df(this%maxpackages, 'WATER MOVER')
    !
    ! -- setup the budget object
    call this%mvr_setup_budobj()
    !
    ! -- Return
    return
  end subroutine mvr_ar

  subroutine mvr_rp(this)
! ******************************************************************************
! mvr_rp -- Read and Prepare
! Subroutine: (1) read itmp
!             (2) read new boundaries if itmp>0
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use TdisModule, only: kper, nper
    use SimModule, only: ustop, store_error, store_error_unit, count_errors
    use ArrayHandlersModule, only: ifind
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    integer(I4B) :: i, ierr, nlist, ipos
    integer(I4B) :: ii, jj
    logical :: isfound, endOfBlock
    character(len=LINELENGTH) :: line, errmsg
    character(len=LENMODELNAME) :: mname
    ! -- formats
    character(len=*),parameter :: fmtblkerr = &
      "('Error.  Looking for BEGIN PERIOD iper.  Found ', a, ' instead.')"
    character(len=*),parameter :: fmtlsp = &
      "(1X,/1X,'REUSING ',A,'S FROM LAST STRESS PERIOD')"
    character(len=*), parameter :: fmtnbd = &
      "(1X,/1X,'THE NUMBER OF ACTIVE ',A,'S (',I6, &
       &') IS GREATER THAN MAXIMUM(',I6,')')"
! ------------------------------------------------------------------------------
    !
    ! -- Set ionper to the stress period number for which a new block of data
    !    will be read.
    if(this%inunit == 0) return
    !
    ! -- get stress period data
    if (this%ionper < kper) then
      !
      ! -- get period block
      call this%parser%GetBlock('PERIOD', isfound, ierr, &
                                supportOpenClose=.true.)
      if(isfound) then
        !
        ! -- read ionper and check for increasing period numbers
        call this%read_check_ionper()
      else
        !
        ! -- PERIOD block not found
        if (ierr < 0) then
          ! -- End of file found; data applies for remainder of simulation.
          this%ionper = nper + 1
        else
          ! -- Found invalid block
          write(errmsg, fmtblkerr) adjustl(trim(line))
          call store_error(errmsg)
          call this%parser%StoreErrorUnit()
          call ustop()
        end if
      endif
    end if
    !
    ! -- read data if ionper == kper
    if(this%ionper == kper) then
      write(this%iout, '(/,2x,a,i0)') 'READING WATER MOVERS FOR PERIOD ', kper
      nlist = -1
      i = 1
      !
      ! -- set mname to '' if this is an exchange mover, or to the model name
      if(this%iexgmvr == 0) then
        mname = this%name_model
      else
        mname = ''
      endif
      !
      ! -- Assign a pointer to the package mover object.  The pointer assignment
      !    will happen only the first time
      call this%assign_packagemovers()
      !
      ! -- Read each mover entry
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetCurrentLine(line)
        !
        ! -- Raise error if movers exceeds maxmvr
        if (i > this%maxmvr) then
          write(errmsg,'(4x,a,a)')'****ERROR. MOVERS EXCEED MAXMVR ON LINE: ', &
                                    trim(adjustl(line))
          call store_error(errmsg)
          call this%parser%StoreErrorUnit()
          call ustop()
        endif
        !
        ! -- Process the water mover line (mname = '' if this is an exchange)
        call this%mvr(i)%set(line, this%parser%iuactive, this%iout, mname,     &
                             this%pckMemPaths, this%pakmovers)
        !
        ! -- Echo input
        if(this%iprpak == 1) call this%mvr(i)%echo(this%iout)
        !
        ! -- increment counter
        i = i + 1
      end do
      write(this%iout,'(/,1x,a,1x,i6,/)') 'END OF DATA FOR PERIOD', kper
      nlist = i - 1
      !
      ! -- Set the number of movers for this period to nlist
      this%nmvr = nlist
      write(this%iout, '(4x, i0, a, i0)') this%nmvr,                           &
        ' MOVERS READ FOR PERIOD ', kper
      !
      ! -- Check to make sure all providers and receivers are properly stored
      do i = 1, this%nmvr
        ipos = ifind(this%pckMemPaths, this%mvr(i)%pckNameSrc)
        if(ipos < 1) then
          write(errmsg,'(4x,a,a,a)') 'ERROR. PROVIDER ',                       &
            trim(this%mvr(i)%pckNameSrc), ' NOT LISTED IN PACKAGES BLOCK.'
          call store_error(errmsg)
        endif
        ipos = ifind(this%pckMemPaths, this%mvr(i)%pckNameTgt)
        if(ipos < 1) then
          write(errmsg,'(4x,a,a,a)') 'ERROR. RECEIVER ',                       &
            trim(this%mvr(i)%pckNameTgt), ' NOT LISTED IN PACKAGES BLOCK.'
          call store_error(errmsg)
        endif
      enddo
      if(count_errors() > 0) then
        call this%parser%StoreErrorUnit()
        call ustop()
      endif
      !
      ! -- reset ientries
      do i = 1, this%maxcomb
        this%ientries(i) = 0
      end do
      !
      ! --
      do i = 1, this%nmvr
        ii = ifind(this%pckMemPaths, this%mvr(i)%pckNameSrc)
        jj = ifind(this%pckMemPaths, this%mvr(i)%pckNameTgt)
        ipos = (ii - 1) * this%maxpackages + jj
        this%ientries(ipos) = this%ientries(ipos) + 1
      end do
    else
      write(this%iout, fmtlsp) 'MVR'
      !
    endif
    !
    ! -- return
    return
  end subroutine mvr_rp

  subroutine mvr_ad(this)
! ******************************************************************************
! mvr_ad -- Advance mover
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- locals
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    do i = 1, this%nmvr
      call this%mvr(i)%advance()
    enddo
    !
    ! -- Return
    return
  end subroutine mvr_ad

  subroutine mvr_fc(this)
! ******************************************************************************
! mvr_fc -- Calculate qfrommvr as a function of qtomvr
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- locals
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    do i = 1, this%nmvr
      call this%mvr(i)%fc()
    enddo
    !
    ! -- Return
    return
  end subroutine mvr_fc

  subroutine mvr_cc(this, innertot, kiter, iend, icnvgmod, cpak, ipak, dpak)
! ******************************************************************************
! mvr_cc -- extra convergence check for mover
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfMvrType) :: this
    integer(I4B),intent(in) :: innertot
    integer(I4B),intent(in) :: kiter
    integer(I4B),intent(in) :: iend
    integer(I4B),intent(in) :: icnvgmod
    character(len=LENPAKLOC), intent(inout) :: cpak
    integer(I4B), intent(inout) :: ipak
    real(DP), intent(inout) :: dpak
    ! -- local
    ! -- formats
    character(len=*),parameter :: fmtmvrcnvg = &
      "(/,1x,'MOVER PACKAGE REQUIRES AT LEAST TWO OUTER ITERATIONS. CONVERGE &
      &FLAG HAS BEEN RESET TO FALSE.')"
! ------------------------------------------------------------------------------
    !
    ! -- If there are active movers, then at least 2 outers required
    if (this%nmvr > 0) then
      if (icnvgmod == 1 .and. kiter == 1) then
        dpak = DNODATA
        cpak = trim(this%packName)
        write(this%iout, fmtmvrcnvg)
      endif
    endif
    !
    ! -- return
    return
  end subroutine mvr_cc
  
  subroutine mvr_bd(this, icbcfl, ibudfl, isuppress_output)
! ******************************************************************************
! mvr_bd -- Write mover terms to listing file
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use TdisModule, only : kstp, kper, delt, pertim, totim
    use InputOutputModule, only: ubdsv06, ubdsvd
    ! -- dummy
    class(GwfMvrType) :: this
    integer(I4B), intent(in) :: icbcfl
    integer(I4B), intent(in) :: ibudfl
    integer(I4B), intent(in) :: isuppress_output
    ! -- locals
    integer(I4B) :: ibinun
    ! -- formats
    character(len=*), parameter :: fmttkk = &
      "(1X,/1X,A,'   PERIOD ',I0,'   STEP ',I0)"
! ------------------------------------------------------------------------------
    !
    ! -- Write the flow table
    if (ibudfl /= 0 .and. this%iprflow /= 0 .and. isuppress_output == 0) then
      call this%mvr_print_outputtab()
    end if
    !
    ! -- fill the budget object
    call this%mvr_fill_budobj()
    !
    ! -- write the flows from the budobj
    ibinun = 0
    if(this%ibudgetout /= 0) then
      ibinun = this%ibudgetout
    end if
    if(icbcfl == 0) ibinun = 0
    if (isuppress_output /= 0) ibinun = 0
    if (ibinun > 0) then
      call this%budobj%save_flows(this%dis, ibinun, kstp, kper, delt, &
                        pertim, totim, this%iout)
    end if
    !
    ! -- Return
    return
  end subroutine mvr_bd

  subroutine mvr_ot(this)
! ******************************************************************************
! mvr_ot -- Write mover budget to listing file
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use TdisModule, only: kstp, kper, delt
    use ArrayHandlersModule, only: ifind, expandarray
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- locals
    character(len=LENMEMPATH) :: pckMemPath
    integer(I4B) :: i, j
    real(DP), allocatable, dimension(:) :: ratin, ratout
! ------------------------------------------------------------------------------
    !
    ! -- Allocate and initialize ratin/ratout
    allocate(ratin(this%maxpackages), ratout(this%maxpackages))
    do j = 1, this%maxpackages
      ratin(j) = DZERO
      ratout(j) = DZERO
    enddo
    !
    ! -- Accumulate the rates
    do i = 1, this%nmvr
      do j = 1, this%maxpackages
        if(this%pckMemPaths(j) == this%mvr(i)%pckNameSrc) then
          ratin(j) = ratin(j) + this%mvr(i)%qpactual
        endif
        if(this%pckMemPaths(j) == this%mvr(i)%pckNameTgt) then
          ratout(j) = ratout(j) + this%mvr(i)%qpactual
        endif
      enddo
    enddo
    !
    ! -- Send rates to budget object
    call this%budget%reset()
    do j = 1, this%maxpackages
      if((this%iexgmvr) == 1) then
        pckMemPath = this%pckMemPaths(j)
      else
        pckMemPath = this%paknames(j)
      endif
      call this%budget%addentry(ratin(j), ratout(j), delt, pckMemPath)
    enddo
    !
    ! -- Write the budget
    call this%budget%budget_ot(kstp, kper, this%iout)
    !
    ! -- Deallocate
    deallocate(ratin, ratout)
    !
    ! -- Output mvr budget
    !    Not using budobj write_table here because it would result
    !    in a table that has one entry.  A custom table looks
    !    better here with a row for each package.
    !call this%budobj%write_budtable(kstp, kper, this%iout)
    !
    ! -- Return
    return
  end subroutine mvr_ot

  subroutine mvr_da(this)
! ******************************************************************************
! mvr_da -- deallocate
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DONE
    use MemoryManagerModule, only: mem_deallocate
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- Arrays
    if (this%inunit > 0) then
      call mem_deallocate(this%ientries)
      deallocate(this%mvr)
      deallocate(this%pckMemPaths)
      deallocate(this%paknames)
      deallocate(this%pakmovers)
      !
      ! -- budget object
      call this%budget%budget_da()
      deallocate(this%budget)
      !
      ! -- budobj
      call this%budobj%budgetobject_da()
      deallocate(this%budobj)
      nullify(this%budobj)
      !
      ! -- output table object
      if (associated(this%outputtab)) then
        call this%outputtab%table_da()
        deallocate(this%outputtab)
        nullify(this%outputtab)
      end if
    endif
    !
    ! -- Scalars
    call mem_deallocate(this%ibudgetout)
    call mem_deallocate(this%maxmvr)
    call mem_deallocate(this%maxpackages)
    call mem_deallocate(this%maxcomb)
    call mem_deallocate(this%nmvr)
    call mem_deallocate(this%iexgmvr)
    call mem_deallocate(this%imodelnames)
    !
    ! -- deallocate scalars in NumericalPackageType
    call this%NumericalPackageType%da()
    !
    ! -- Return
    return
  end subroutine mvr_da

  subroutine read_options(this)
! ******************************************************************************
! read_options
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH, DZERO, DONE
    use OpenSpecModule, only: access, form
    use SimModule, only: ustop, store_error, store_error_unit
    use InputOutputModule, only: urword, getunit, openfile
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg
    character(len=MAXCHARLEN) :: fname, keyword
    integer(I4B) :: ierr
    logical :: isfound, endOfBlock
    ! -- formats
    character(len=*),parameter :: fmtmvrbin = &
      "(4x, 'MVR ', 1x, a, 1x, ' WILL BE SAVED TO FILE: ', a, /4x, 'OPENED ON UNIT: ', I7)"
! ------------------------------------------------------------------------------
    !
    ! -- get options block
    call this%parser%GetBlock('OPTIONS', isfound, ierr, &
      supportOpenClose=.true., blockRequired=.false.)
    !
    ! -- parse options block if detected
    if (isfound) then
      write(this%iout,'(1x,a)')'PROCESSING MVR OPTIONS'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case('BUDGET')
            call this%parser%GetStringCaps(keyword)
            if (keyword == 'FILEOUT') then
              call this%parser%GetString(fname)
              this%ibudgetout = getunit()
              call openfile(this%ibudgetout, this%iout, fname, 'DATA(BINARY)',  &
                            form, access, 'REPLACE')
              write(this%iout,fmtmvrbin) 'BUDGET', fname, this%ibudgetout
            else
              call store_error('OPTIONAL BUDGET KEYWORD MUST BE FOLLOWED BY FILEOUT')
            end if
          case ('PRINT_INPUT')
            this%iprpak = 1
            write(this%iout,'(4x,a)') 'WATER MOVER INPUT '//                   &
              'WILL BE PRINTED TO LIST FILE.'
          case ('PRINT_FLOWS')
            this%iprflow = 1
            write(this%iout,'(4x,a)') 'LISTS OF WATER MOVER FLOWS '//          &
              'WILL BE PRINTED TO LIST FILE.'
          case ('MODELNAMES')
            this%imodelnames = 1
            write(this%iout,'(4x,a)') 'ALL PACKAGE NAMES ARE PRECEDED '//      &
              'BY THE NAME OF THE MODEL CONTAINING THE PACKAGE.'
            if (this%iexgmvr == 0) then
              write(errmsg,'(4x,a,a)')                                         &
                '****ERROR. MODELNAMES CANNOT BE SPECIFIED UNLESS THE ' //     &
                'MOVER PACKAGE IS FOR AN EXCHANGE.'
              call store_error(errmsg)
              call this%parser%StoreErrorUnit()
              call ustop()
            endif
          case default
            write(errmsg,'(4x,a,a)')'****ERROR. UNKNOWN MVR OPTION: ',         &
                                     trim(keyword)
            call store_error(errmsg)
            call this%parser%StoreErrorUnit()
            call ustop()
        end select
      end do
      write(this%iout,'(1x,a)')'END OF MVR OPTIONS'
    end if
    !
    ! -- Return
    return
    end subroutine read_options

  subroutine check_options(this)
! ******************************************************************************
! check_options
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use SimModule, only: ustop, store_error, store_error_unit
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! -- Check if not exchange mover but model names are specified
    if (this%iexgmvr == 0 .and. this%imodelnames == 1) then
      write(errmsg,'(4x,a,a)')                                                 &
        '****ERROR. MODELNAMES CANNOT BE SPECIFIED UNLESS THE ' //             &
        'MOVER PACKAGE IS FOR AN EXCHANGE.'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Check if exchange mover but model names not specified
    if (this%iexgmvr /= 0 .and. this%imodelnames == 0) then
      write(errmsg,'(4x,a,a)')                                                 &
        '****ERROR. MODELNAMES OPTION MUST BE SPECIFIED BECAUSE ' //           &
        'MOVER PACKAGE IS FOR AN EXCHANGE.'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Return
    return
  end subroutine check_options

  subroutine read_dimensions(this)
! ******************************************************************************
! read_dimensions -- Read the dimensions for this package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use SimModule, only: ustop, store_error, count_errors, store_error_unit
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: errmsg, keyword
    integer(I4B) :: ierr
    logical :: isfound, endOfBlock
    integer(I4B) :: i
    integer(I4B) :: j
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- get dimensions block
    call this%parser%GetBlock('DIMENSIONS', isfound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse dimensions block if detected
    if (isfound) then
      write(this%iout,'(/1x,a)')'PROCESSING MVR DIMENSIONS'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case ('MAXMVR')
            this%maxmvr = this%parser%GetInteger()
            write(this%iout,'(4x,a,i0)')'MAXMVR = ', this%maxmvr
          case ('MAXPACKAGES')
            this%maxpackages  = this%parser%GetInteger()
            write(this%iout,'(4x,a,i0)')'MAXPACKAGES = ', this%maxpackages
          case default
            write(errmsg,'(4x,a,a)')                                           &
              '****ERROR. UNKNOWN MVR DIMENSION: ', trim(keyword)
            call store_error(errmsg)
            call this%parser%StoreErrorUnit()
            call ustop()
        end select
      end do
      write(this%iout,'(1x,a)')'END OF MVR DIMENSIONS'
    else
      call store_error('ERROR.  REQUIRED DIMENSIONS BLOCK NOT FOUND.')
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- calculate maximum number of combinations
    this%maxcomb = 0
    do i = 1, this%maxpackages
      do j = 1, this%maxpackages
        this%maxcomb = this%maxcomb + 1
      end do
    end do
    !
    ! -- verify dimensions were set
    if(this%maxmvr < 0) then
      write(errmsg, '(1x,a)') &
        'ERROR.  MAXMVR WAS NOT SPECIFIED OR WAS SPECIFIED INCORRECTLY.'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    if(this%maxpackages < 0) then
      write(errmsg, '(1x,a)') &
        'ERROR.  MAXPACKAGES WAS NOT SPECIFIED OR WAS SPECIFIED INCORRECTLY.'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- return
    return
  end subroutine read_dimensions

  subroutine read_packages(this)
! ******************************************************************************
! read_packages -- Read the packages that will be managed by this mover
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use MemoryHelperModule, only: create_mem_path
    use SimModule, only: ustop, store_error, count_errors, store_error_unit
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: errmsg, word, word1, word2
    integer(I4B) :: lloc, ierr
    integer(I4B) :: npak
    logical :: isfound, endOfBlock
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- get packages block
    call this%parser%GetBlock('PACKAGES', isfound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse packages block
    if (isfound) then
      write(this%iout,'(/1x,a)')'PROCESSING MVR PACKAGES'
      npak = 0
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(word1)
        lloc = 1
        npak = npak + 1
        if (npak > this%maxpackages) then
          call store_error('ERROR.  MAXPACKAGES NOT SET LARGE ENOUGH.')
          call this%parser%StoreErrorUnit()
          call ustop()
        endif
        if(this%iexgmvr == 0) then
          this%pckMemPaths(npak) = create_mem_path(this%name_model, word1)
          word = word1
        else
          this%pckMemPaths(npak) = trim(word1)
          call this%parser%GetStringCaps(word2)
          this%pckMemPaths(npak) = create_mem_path(this%pckMemPaths(npak), word2)
          word = word2
        endif
        this%paknames(npak) = trim(word)
        write(this%iout,'(3x,a,a)')'INCLUDING PACKAGE: ',                  &
          trim(this%pckMemPaths(npak))
      end do
      write(this%iout,'(1x,a)')'END OF MVR PACKAGES'
    else
      call store_error('ERROR.  REQUIRED PACKAGES BLOCK NOT FOUND.')
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- Check to make sure npak = this%maxpackages
    if(npak /= this%maxpackages) then
      write(errmsg, '(a, i0, a, i0, a)')                                        &
        'ERROR.  NUMBER OF PACKAGES (', npak, ') DOES NOT EQUAL ' //            &
        'MAXPACKAGES (', this%maxpackages, ').'
      call store_error(errmsg)
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- return
    return
  end subroutine read_packages

  subroutine check_packages(this)
! ******************************************************************************
! check_packages -- check to make sure packages have mover activated
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LINELENGTH
    use MemoryManagerModule, only: mem_setptr
    use SimModule, only: ustop, store_error, count_errors, store_error_unit
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: errmsg
    integer(I4B) :: i
    integer(I4B), pointer :: imover_ptr
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- Check to make sure mover is activated for each package
    do i = 1, size(this%pckMemPaths)
      imover_ptr => null()
      call mem_setptr(imover_ptr, 'IMOVER', trim(this%pckMemPaths(i)))
      if (imover_ptr == 0) then
        write(errmsg, '(a, a, a)') &
                          'ERROR.  MODEL AND PACKAGE "', &
                          trim(this%pckMemPaths(i)), &
                          '" DOES NOT HAVE MOVER SPECIFIED IN OPTIONS BLOCK.'
        call store_error(errmsg)
      end if
    end do
    !
    ! -- Terminate if errors detected.
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- return
    return
  end subroutine check_packages

  subroutine assign_packagemovers(this)
! ******************************************************************************
! assign_packagemovers -- assign pointer to each package's packagemover object
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use PackageMoverModule, only: set_packagemover_pointer
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    integer(I4B) :: i
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- Assign the package mover pointer if it hasn't been assigned yet
    do i = 1, size(this%pckMemPaths)
      if (this%pakmovers(i)%memoryPath == '') then
        call set_packagemover_pointer(this%pakmovers(i), &
                                      trim(this%pckMemPaths(i)))
      end if
    end do
    !
    ! -- return
    return
  end subroutine assign_packagemovers

  subroutine allocate_scalars(this)
! ******************************************************************************
! allocate_scalars
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DONE
    use MemoryManagerModule, only: mem_allocate
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- allocate scalars in NumericalPackageType
    call this%NumericalPackageType%allocate_scalars()
    !
    ! -- Allocate
    call mem_allocate(this%ibudgetout, 'IBUDGETOUT', this%memoryPath)
    call mem_allocate(this%maxmvr, 'MAXMVR', this%memoryPath)
    call mem_allocate(this%maxpackages, 'MAXPACKAGES', this%memoryPath)
    call mem_allocate(this%maxcomb, 'MAXCOMB', this%memoryPath)
    call mem_allocate(this%nmvr, 'NMVR', this%memoryPath)
    call mem_allocate(this%iexgmvr, 'IEXGMVR', this%memoryPath)
    call mem_allocate(this%imodelnames, 'IMODELNAMES', this%memoryPath)
    !
    ! -- Initialize
    this%ibudgetout = 0
    this%maxmvr = -1
    this%maxpackages = -1
    this%maxcomb = 0
    this%nmvr = 0
    this%iexgmvr = 0
    this%imodelnames = 0
    !
    ! -- Return
    return
  end subroutine allocate_scalars

  subroutine allocate_arrays(this)
! ******************************************************************************
! allocate_arrays
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use MemoryManagerModule, only: mem_allocate
    use ConstantsModule, only: DZERO
    use PackageMoverModule, only: nulllify_packagemover_pointer
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- Allocate
    allocate(this%mvr(this%maxmvr))
    allocate(this%pckMemPaths(this%maxpackages))
    allocate(this%paknames(this%maxpackages))
    allocate(this%pakmovers(this%maxpackages))
    !
    ! -- nullify the pakmovers
    do i = 1, this%maxpackages
      call nulllify_packagemover_pointer(this%pakmovers(i))
    end do
    !
    ! -- allocate the object and assign values to object variables
    call mem_allocate(this%ientries, this%maxcomb, 'IENTRIES', this%memoryPath)
    !
    ! -- setup the output table
    call this%mvr_setup_outputtab()
    !
    ! -- Return
    return
  end subroutine allocate_arrays

  subroutine mvr_setup_budobj(this)
! ******************************************************************************
! mvr_setup_budobj -- Set up the budget object that stores all the mvr flows
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: LENBUDTXT
    use MemoryHelperModule, only: split_mem_path
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
    integer(I4B) :: nbudterm
    integer(I4B) :: ncv
    integer(I4B) :: i
    integer(I4B) :: j
    integer(I4B) :: naux
    character (len=LENMODELNAME) :: modelname1, modelname2
    character (len=LENPACKAGENAME) :: packagename1, packagename2
    integer(I4B) :: maxlist
    integer(I4B) :: idx
    character(len=LENBUDTXT) :: text
! ------------------------------------------------------------------------------
    !
    ! -- Determine the number of mover budget terms. These are fixed for 
    !    the simulation and cannot change.  A separate term is required
    !    for each possible provider/receiver combination.
    nbudterm = 0
    do i = 1, this%maxpackages
      do j = 1, this%maxpackages
        nbudterm = nbudterm + 1
      end do
    end do
    !
    ! -- Number of control volumes is set to be 0, because there aren't
    !    any for the mover
    ncv = 0
    !
    ! -- set up budobj
    call budgetobject_cr(this%budobj, 'WATER MOVER')
    call this%budobj%budgetobject_df(ncv, nbudterm, 0, 0)
    idx = 0
    !
    ! -- Go through and set up each budget term
    text = '      MOVER-FLOW'
    maxlist = this%maxmvr
    naux = 0
    do i = 1, this%maxpackages
      
      call split_mem_path(this%pckMemPaths(i), modelname1, packagename1)
      
      do j = 1, this%maxpackages  
        
        idx = idx + 1
        call split_mem_path(this%pckMemPaths(j), modelname2, packagename2)
        call this%budobj%budterm(idx)%initialize(text, &
                                                 modelname1, &
                                                 packagename1, &
                                                 modelname2, &
                                                 packagename2, &
                                                 maxlist, .false., .false., &
                                                 naux)
      end do
    end do
    !
    ! -- return
    return
  end subroutine mvr_setup_budobj

  subroutine mvr_fill_budobj(this)
! ******************************************************************************
! mvr_fill_budobj -- copy flow terms into this%budobj
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfMvrType) :: this
    ! -- local
    integer(I4B) :: idx
    integer(I4B) :: i
    integer(I4B) :: j
    integer(I4B) :: n, n1, n2
    integer(I4B) :: ipos
    integer(I4B) :: ival
    integer(I4B) :: nitems
    integer(I4B) :: lloc
    integer(I4B) :: istart
    integer(I4B) :: istop
    real(DP) :: rval
    character (len=LENMODELNAME) :: modelname1, modelname2
    character (len=LENPACKAGENAME) :: packagename1, packagename2
    character (len=LENMEMPATH) :: pckMemPathsDummy
    real(DP) :: q
    ! -- formats
! -----------------------------------------------------------------------------
    !
    ! -- initialize counter
    idx = 0


    do i = 1, this%maxpackages
      ! -- Retrieve modelname1 and packagename1
      lloc = 1
      call urword(this%pckMemPaths(i), lloc, istart, istop, 1, ival, rval, -1, -1)
      pckMemPathsDummy = this%pckMemPaths(i)
      modelname1 = pckMemPathsDummy(istart:istop)
      call urword(this%pckMemPaths(i), lloc, istart, istop, 1, ival, rval, -1, -1)
      pckMemPathsDummy = this%pckMemPaths(i)
      packagename1 = pckMemPathsDummy(istart:istop)
      do j = 1, this%maxpackages
        ! -- Retrieve modelname2 and packagename2
        lloc = 1
        call urword(this%pckMemPaths(j), lloc, istart, istop, 1, ival, rval, -1, -1)
        pckMemPathsDummy = this%pckMemPaths(j)
        modelname2 = pckMemPathsDummy(istart:istop)
        call urword(this%pckMemPaths(j), lloc, istart, istop, 1, ival, rval, -1, -1)
        pckMemPathsDummy = this%pckMemPaths(j)
        packagename2 = pckMemPathsDummy(istart:istop)
        ipos = (i - 1) * this%maxpackages + j
        nitems = this%ientries(ipos)
        !
        ! -- nitems is the number of mover connections for this
        !    model-package / model-package combination.  Cycle if none.
        idx = idx + 1
        call this%budobj%budterm(idx)%reset(nitems)
        if (nitems < 1) cycle
        do n = 1, this%nmvr
          !
          ! -- pname1 is provider, pname2 is receiver
          !    flow is always negative because it is coming from provider
          if(this%pckMemPaths(i) == this%mvr(n)%pckNameSrc) then
            if(this%pckMemPaths(j) == this%mvr(n)%pckNameTgt) then
              !
              ! -- set q to qpactual
              q = -this%mvr(n)%qpactual
              !
              ! -- map from irch1 to feature (needed for lake to map outlet to lake number)
              n1 = this%mvr(n)%iRchNrSrc
              n1 = this%pakmovers(i)%iprmap(n1)
              !
              ! -- set receiver id to irch2
              n2 = this%mvr(n)%iRchNrTgt
              !
              ! -- check record into budget object
              call this%budobj%budterm(idx)%update_term(n1, n2, q)
            end if
          end if
        end do
      end do
    end do
    !
    ! --Terms are filled, now accumulate them for this time step
    call this%budobj%accumulate_terms()
    !
    ! -- return
    return
  end subroutine mvr_fill_budobj

  subroutine mvr_setup_outputtab(this)
! ******************************************************************************
! mvr_setup_outputtab -- set up output table
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    character(len=LINELENGTH) :: title
    character(len=LINELENGTH) :: text
    integer(I4B) :: ntabcol
    integer(I4B) :: ilen
! ------------------------------------------------------------------------------
    !
    ! -- allocate and initialize the output table
    if (this%iprflow /= 0) then
      !
      ! -- dimension table
      ntabcol = 7
      !
      ! -- initialize the output table object
      title = 'WATER MOVER PACKAGE (' // trim(this%packName) //     &
              ') FLOW RATES'
      call table_cr(this%outputtab, this%packName, title)
      call this%outputtab%table_df(this%maxmvr, ntabcol, this%iout,            &
                                    transient=.TRUE.)
      text = 'NUMBER'
      call this%outputtab%initialize_column(text, 10, alignment=TABCENTER)
      text = 'PROVIDER LOCATION'
      ilen = LENMODELNAME+LENPACKAGENAME+1
      call this%outputtab%initialize_column(text, ilen)
      text = 'PROVIDER ID'
      call this%outputtab%initialize_column(text, 10)
      text = 'AVAILABLE RATE'
      call this%outputtab%initialize_column(text, 10)
      text = 'PROVIDED RATE'
      call this%outputtab%initialize_column(text, 10)
      text = 'RECEIVER LOCATION'
      ilen = LENMODELNAME+LENPACKAGENAME+1
      call this%outputtab%initialize_column(text, ilen)
      text = 'RECEIVER ID'
      call this%outputtab%initialize_column(text, 10)
      
    end if
    !
    ! -- return
    return
  end subroutine mvr_setup_outputtab

  subroutine mvr_print_outputtab(this)
! ******************************************************************************
! mvr_setup_outputtab -- set up output table
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- module
    use TdisModule, only: kstp, kper
    ! -- dummy
    class(GwfMvrType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: title
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- set table kstp and kper
    call this%outputtab%set_kstpkper(kstp, kper)
    !
    ! -- Add terms and print the table
    title = 'WATER MOVER PACKAGE (' // trim(this%packName) //     &
            ') FLOW RATES'
    call this%outputtab%set_title(title)
    call this%outputtab%set_maxbound(this%nmvr)
    do i = 1, this%nmvr
      call this%outputtab%add_term(i)
      call this%outputtab%add_term(this%mvr(i)%pckNameSrc)
      call this%outputtab%add_term(this%mvr(i)%iRchNrSrc)
      call this%outputtab%add_term(this%mvr(i)%qavailable)
      call this%outputtab%add_term(this%mvr(i)%qpactual)
      call this%outputtab%add_term(this%mvr(i)%pckNameTgt)
      call this%outputtab%add_term(this%mvr(i)%iRchNrTgt)
    end do
    !
    ! -- return
    return
  end subroutine mvr_print_outputtab

end module
