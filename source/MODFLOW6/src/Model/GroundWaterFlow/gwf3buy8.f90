! Buoyancy Package for representing variable-density groundwater flow
! The BUY Package does not work yet with the NPF XT3D option

  
module GwfBuyModule
  
  use KindModule,             only: DP, I4B
  use SimModule,              only: ustop, store_error, count_errors,          &
                                    store_error_unit
  use MemoryManagerModule,    only: mem_allocate, mem_reallocate,              &
                                    mem_deallocate
  use ConstantsModule,        only: DHALF, DZERO, DONE, LENMODELNAME,          &
                                    LENAUXNAME, DHNOFLO, MAXCHARLEN, LINELENGTH
  use NumericalPackageModule, only: NumericalPackageType
  use BaseDisModule,          only: DisBaseType
  use GwfNpfModule,           only: GwfNpfType
  
  implicit none

  private
  public :: GwfBuyType
  public :: buy_cr
  
  type :: ConcentrationPointer
    real(DP), dimension(:), pointer             :: conc       => null()         ! pointer to concentration array
    integer(I4B), dimension(:), pointer         :: icbund     => null()         ! store pointer to gwt ibound array
  end type ConcentrationPointer

  type, extends(NumericalPackageType) :: GwfBuyType
    type(GwfNpfType), pointer                   :: npf        => null()         ! npf object
    integer(I4B), pointer                       :: ioutdense  => null()         ! unit number for saving density
    integer(I4B), pointer                       :: iform      => null()         ! formulation: 0 freshwater head, 1 hydraulic head, 2 hh rhs
    integer(I4B), pointer                       :: ireadelev  => null()         ! if 1 then elev has been allocated and filled
    integer(I4B), pointer                       :: ireadconcbuy => null()       ! if 1 then dense has been read from this buy input file
    integer(I4B), pointer                       :: iconcset   => null()         ! if 1 then conc is pointed to a gwt model%x
    real(DP), pointer                           :: denseref   => null()         ! reference fluid density
    real(DP), dimension(:), pointer, contiguous :: dense      => null()         ! density
    real(DP), dimension(:), pointer, contiguous :: concbuy    => null()         ! concentration array if specified in buy package
    real(DP), dimension(:), pointer, contiguous :: elev       => null()         ! cell center elevation (optional; if not specified, hten use (top+bot)/2)
    integer(I4B), dimension(:), pointer         :: ibound     => null()         ! store pointer to ibound
    
    integer(I4B), pointer                       :: nrhospecies => null()        ! number of species used in equation of state to calculate density
    real(DP), dimension(:), pointer, contiguous :: drhodc      => null()        ! change in density with change in concentration
    real(DP), dimension(:), pointer, contiguous :: crhoref     => null()        ! reference concentration used in equation of state
    real(DP), dimension(:), pointer, contiguous :: ctemp       => null()        ! temporary array of size (nrhospec) to pass to calcdens
    character(len=LENMODELNAME), dimension(:), allocatable :: cmodelname        ! names of gwt models used in equation of state
    character(len=LENAUXNAME), dimension(:), allocatable :: cauxspeciesname     ! names of gwt models used in equation of state

    type(ConcentrationPointer), allocatable, dimension(:) :: modelconc           ! concentration pointer for each transport model
    
  contains    
    procedure :: buy_df
    procedure :: buy_ar
    procedure :: buy_ar_bnd
    procedure :: buy_rp
    procedure :: buy_ad
    procedure :: buy_cf
    procedure :: buy_cf_bnd
    procedure :: buy_fc
    procedure :: buy_bdsav
    procedure :: buy_flowja
    procedure :: buy_da
    procedure, private :: read_dimensions
    procedure, private :: read_packagedata
    procedure, private :: calcbuy
    procedure, private :: calchhterms
    procedure, private :: buy_calcdens
    procedure, private :: buy_calcelev
    procedure :: allocate_scalars
    procedure, private :: allocate_arrays
    procedure, private :: read_options
    procedure :: set_concentration_pointer
  end type GwfBuyType
  
  contains
  
  function calcdens(denseref, drhodc, crhoref, conc) result(dense)
! ******************************************************************************
! calcdens -- generic function to calculate fluid density from concentration
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    real(DP), intent(in) :: denseref
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(in) :: conc
    ! -- return
    real(DP) :: dense
    ! -- local
    integer(I4B) :: nrhospec
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    nrhospec = size(drhodc)
    dense = denseref
    do i = 1, nrhospec
      dense = dense + drhodc(i) * (conc(i) - crhoref(i))
    end do
    !
    ! -- return
    return
  end function calcdens

  subroutine buy_cr(buyobj, name_model, inunit, iout)
! ******************************************************************************
! buy_cr -- Create a new BUY object
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    type(GwfBuyType), pointer :: buyobj
    character(len=*), intent(in) :: name_model
    integer(I4B), intent(in) :: inunit
    integer(I4B), intent(in) :: iout
! ------------------------------------------------------------------------------
    !
    ! -- Create the object
    allocate(buyobj)
    !
    ! -- create name and memory path
    call buyobj%set_names(1, name_model, 'BUY', 'BUY')
    !
    ! -- Allocate scalars
    call buyobj%allocate_scalars()
    !
    ! -- Set variables
    buyobj%inunit = inunit
    buyobj%iout = iout
    !
    ! -- Initialize block parser
    call buyobj%parser%Initialize(buyobj%inunit, buyobj%iout)
    !
    ! -- Return
    return
  end subroutine buy_cr

  subroutine buy_df(this, dis)
! ******************************************************************************
! buy_df -- Allocate and Read
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
    class(DisBaseType), pointer, intent(in) :: dis
    ! -- local
    ! -- formats
    character(len=*), parameter :: fmtbuy =                                    &
      "(1x,/1x,'BUY -- BUOYANCY PACKAGE, VERSION 1, 5/16/2018',                &
      &' INPUT READ FROM UNIT ', i0, //)"
! ------------------------------------------------------------------------------
    !
    ! --print a message identifying the buoyancy package.
    write(this%iout, fmtbuy) this%inunit
    !
    ! -- store pointers to arguments that were passed in
    this%dis => dis
    !
    ! -- Read buoyancy options
    call this%read_options()
    !
    ! -- Read buoyancy dimensions
    call this%read_dimensions()
    !
    ! -- Allocate arrays
    call this%allocate_arrays(dis%nodes)
    !
    ! -- Read buoyancy packagedata
    call this%read_packagedata()
    !
    ! -- Return
    return
  end subroutine buy_df

  subroutine buy_ar(this, npf, ibound)
! ******************************************************************************
! buy_ar -- Allocate and Read
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
    type(GwfNpfType), pointer, intent(in) :: npf
    integer(I4B), dimension(:), pointer :: ibound
    ! -- local
    ! -- formats
! ------------------------------------------------------------------------------
    !
    ! -- store pointers to arguments that were passed in
    this%npf     => npf
    this%ibound  => ibound
    !
    ! -- Ensure NPF XT3D is not on
    if (this%npf%ixt3d /= 0) then
      call store_error('Error in model ' // trim(this%name_model) // &
        '.  The XT3D option cannot be used with the BUY Package.')
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- Calculate cell elevations
    call this%buy_calcelev()
    !
    ! -- Return
    return
  end subroutine buy_ar

  subroutine buy_ar_bnd(this, packobj, hnew)
! ******************************************************************************
! buy_ar_bnd -- buoyancy ar_bnd routine to activate density in packages.
!   This routine is called from gwf_ar() as it goes through each package.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    use LakModule, only: LakType
    use SfrModule, only: SfrType
    use MawModule, only: MawType
    ! -- dummy
    class(GwfBuyType) :: this
    class(BndType), pointer :: packobj
    real(DP), intent(in), dimension(:) :: hnew
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- Add density terms based on boundary package type
    select case (packobj%filtyp)
      case('LAK')
        !
        ! -- activate density for lake package
        select type(packobj)
        type is (LakType)
          call packobj%lak_activate_density()
        end select
        
      case('SFR')
        !
        ! -- activate density for sfr package
        select type(packobj)
        type is (SfrType)
          call packobj%sfr_activate_density()
        end select
        
      case('MAW')
        !
        ! -- activate density for maw package
        select type(packobj)
        type is (MawType)
          call packobj%maw_activate_density()
        end select
        
      case default
        !
        ! -- nothing
    end select
    !
    ! -- Return
    return
  end subroutine buy_ar_bnd
  
  subroutine buy_rp(this)
! ******************************************************************************
! buy_rp -- Check for new buy period data
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use TdisModule, only: kstp, kper
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg
    integer(I4B) :: i
    ! -- formats
    character(len=*),parameter :: fmtc = &
      "('BUOYANCY PACKAGE DOES NOT HAVE HAVE A CONCENTRATION SET &
       &FOR SPECIES ',i0,'. ONE OR MORE MODEL NAMES MAY BE SPECIFIED &
       &INCORRECTLY IN THE PACKAGEDATA BLOCK OR A GWF-GWT EXCHANGE MAY NEED &
       &TO BE ACTIVATED.')"
! ------------------------------------------------------------------------------
    !
    ! -- Check to make sure all concentration pointers have been set
    if (kstp * kper == 1) then
      do i = 1, this%nrhospecies
        if (.not. associated(this%modelconc(i)%conc)) then
          write(errmsg, fmtc) i
          call store_error(errmsg)
        end if
      end do
      if (count_errors() > 0) then
        call this%parser%StoreErrorUnit()
        call ustop()
      end if
    end if
    !
    ! -- return
    return
  end subroutine buy_rp

  subroutine buy_ad(this)
! ******************************************************************************
! buy_ad -- Advance
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- update density using the last concentration
    call this%buy_calcdens()
    !
    ! -- Return
    return
  end subroutine buy_ad
  
  subroutine buy_cf(this, kiter)
! ******************************************************************************
! buy_cf -- Fill coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B) :: kiter
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- Recalculate the elev array for this iteration
    if (this%ireadelev == 0) then
      if (this%iform == 1 .or. this%iform == 2) then
        call this%buy_calcelev()
      end if
    end if
    !
    ! -- Return
    return
  end subroutine buy_cf
  
  subroutine buy_cf_bnd(this, packobj, hnew) !, hcof, rhs, auxnam, auxvar)
! ******************************************************************************
! buy_cf_bnd -- Fill coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    ! -- dummy
    class(GwfBuyType) :: this
    class(BndType), pointer :: packobj
    real(DP), intent(in), dimension(:) :: hnew
    ! -- local
    integer(I4B) :: i, j
    integer(I4B) :: n, locdense, locelev
    integer(I4B), dimension(:), allocatable :: locconc
! ------------------------------------------------------------------------------
    !
    ! -- Return if freshwater head formulation; all boundary heads must be
    !    entered as freshwater equivalents
    if (this%iform == 0) return
    !
    ! -- initialize
    locdense = 0
    locelev = 0
    allocate(locconc(this%nrhospecies))
    locconc(:) = 0
    !
    ! -- Add buoyancy terms for head-dependent boundaries
    do n = 1, packobj%naux
      if (packobj%auxname(n) == 'DENSITY') then
        locdense = n
      else if (packobj%auxname(n) == 'ELEVATION') then
        locelev = n
      end if
    end do
    !
    ! -- find aux columns for concentrations that affect density
    do i = 1, this%nrhospecies
      locconc(i) = 0
      do j = 1, packobj%naux
        if (this%cauxspeciesname(i) == packobj%auxname(j)) then
          locconc(i) = j
          exit
        end if
      end do
      if (locconc(i) == 0) then
        ! -- one not found, so don't use and mark all as 0
        locconc(:) = 0
        exit
      end if
    end do
    !
    ! -- Add density terms based on boundary package type
    select case (packobj%filtyp)
      case('GHB')
        !
        ! -- general head boundary
        call buy_cf_ghb(packobj, hnew, this%dense, this%elev, this%denseref,   &
                        locelev, locdense, locconc, this%drhodc, this%crhoref, &
                        this%ctemp, this%iform)
      case('RIV')
        !
        ! -- river
        call buy_cf_riv(packobj, hnew, this%dense, this%elev, this%denseref,   &
                        locelev, locdense, locconc, this%drhodc, this%crhoref, &
                        this%ctemp, this%iform)
      case('DRN')
        !
        ! -- drain
        call buy_cf_drn(packobj, hnew, this%dense, this%denseref)
      case('LAK')
        !
        ! -- lake
        call buy_cf_lak(packobj, hnew, this%dense, this%elev, this%denseref,   &
                        locdense, locconc, this%drhodc, this%crhoref,          &
                        this%ctemp, this%iform)
      case('SFR')
        !
        ! -- sfr
        call buy_cf_sfr(packobj, hnew, this%dense, this%elev, this%denseref,   &
                        locdense, locconc, this%drhodc, this%crhoref,          &
                        this%ctemp, this%iform)
      case('MAW')
        !
        ! -- maw
        call buy_cf_maw(packobj, hnew, this%dense, this%elev, this%denseref,   &
                        locdense, locconc, this%drhodc, this%crhoref,          &
                        this%ctemp, this%iform)
      case default
        !
        ! -- nothing
    end select
    !
    ! -- deallocate
    deallocate(locconc)
    !
    ! -- Return
    return
  end subroutine buy_cf_bnd
  
  function get_bnd_density(n, locdense, locconc, denseref, drhodc, crhoref,    &
                           ctemp, auxvar) result (densebnd)
! ******************************************************************************
! get_bnd_density -- Return the density of the boundary package using one of
!   several different options in the following order of priority:
!     1. Assign as aux variable in column with name 'DENSITY'
!     2. Calculate using equation of state and nrhospecies aux columns
!     3. If neither of those, then assign as denseref
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    integer(I4B), intent(in) :: n
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), intent(in) :: denseref
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    real(DP), dimension(:, :), intent(in) :: auxvar
    ! -- return
    real(DP) :: densebnd
    ! -- local
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- assign boundary density based on one of three options
    if (locdense > 0) then
      ! -- assign density to an aux column named 'DENSITY'
      densebnd = auxvar(locdense, n)
    else if (locconc(1) > 0) then
      ! -- calculate density using one or more concentration auxcolumns
      do i = 1, size(locconc)
        ctemp(i) = DZERO
        if (locconc(i) > 0) then
          ctemp(i) = auxvar(locconc(i), n)
        end if
      end do
      densebnd = calcdens(denseref, drhodc, crhoref, ctemp)
    else
      ! -- neither of the above, so assign as denseref
      densebnd = denseref
    end if
    !
    ! -- return
    return
  end function get_bnd_density
  
  subroutine buy_cf_ghb(packobj, hnew, dense, elev, denseref, locelev,         &
                        locdense, locconc, drhodc, crhoref, ctemp,             &
                        iform)
! ******************************************************************************
! buy_cf_ghb -- Fill ghb coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in), dimension(:) :: elev
    real(DP), intent(in) :: denseref
    integer(I4B), intent(in) :: locelev
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    integer(I4B), intent(in) :: iform
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: denseghb
    real(DP) :: elevghb
    real(DP) :: hghb
    real(DP) :: cond
    real(DP) :: hcofterm, rhsterm
! ------------------------------------------------------------------------------
    !
    ! -- Process density terms for each GHB
    do n = 1, packobj%nbound
      node = packobj%nodelist(n)
      if (packobj%ibound(node) <= 0) cycle
      !
      ! -- density
      denseghb = get_bnd_density(n, locdense, locconc, denseref, &
                                   drhodc, crhoref, ctemp, packobj%auxvar)
      !
      ! -- elevation
      elevghb = elev(node)
      if (locelev > 0) elevghb = packobj%auxvar(locelev, n)
      !
      ! -- boundary head and conductance
      hghb = packobj%bound(1, n)
      cond = packobj%bound(2, n)
      !
      ! -- calculate HCOF and RHS terms
      call calc_ghb_hcof_rhs_terms(denseref, denseghb, dense(node),            &
                                   elevghb, elev(node), hghb, hnew(node),      &
                                   cond, iform, rhsterm, hcofterm)      
      packobj%hcof(n) = packobj%hcof(n) + hcofterm
      packobj%rhs(n) = packobj%rhs(n) - rhsterm
      !
    end do
    !
    ! -- Return
    return
  end subroutine buy_cf_ghb
                        
  subroutine calc_ghb_hcof_rhs_terms(denseref, denseghb, densenode, &
                                     elevghb, elevnode, hghb, hnode, &
                                     cond, iform, rhsterm, hcofterm)
! ******************************************************************************
! calc_ghb_hcof_rhs_terms -- Calculate density hcof and rhs terms for ghb 
!   conditions
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    real(DP), intent(in) :: denseref
    real(DP), intent(in) :: denseghb
    real(DP), intent(in) :: densenode
    real(DP), intent(in) :: elevghb
    real(DP), intent(in) :: elevnode
    real(DP), intent(in) :: hghb
    real(DP), intent(in) :: hnode
    real(DP), intent(in) :: cond
    integer(I4B), intent(in) :: iform
    real(DP), intent(inout) :: rhsterm
    real(DP), intent(inout) :: hcofterm
    ! -- local
    real(DP) :: t1, t2
    real(DP) :: avgdense, avgelev
! ------------------------------------------------------------------------------
    !
    ! -- Calculate common terms
    avgdense = DHALF * denseghb + DHALF * densenode
    avgelev = DHALF * elevghb + DHALF * elevnode
    t1 = avgdense / denseref - DONE
    t2 = (denseghb - densenode) / denseref
    !
    ! -- Add hcof terms
    hcofterm = -cond * t1
    if (iform == 2) then
      !
      ! -- this term goes on RHS for iform == 1
      hcofterm = hcofterm + DHALF * cond * t2
    end if
    !
    ! -- Add rhs terms
    rhsterm = cond * t1 * hghb
    rhsterm = rhsterm - cond * t2 * avgelev
    rhsterm = rhsterm + DHALF * cond * t2 * hghb
    if (iform == 1) then
      !
      ! -- this term goes on LHS for iform == 2
      rhsterm = rhsterm + DHALF * cond * t2 * hnode
    end if
    !
    ! -- return
    return
  end subroutine calc_ghb_hcof_rhs_terms
  
  subroutine buy_cf_riv(packobj, hnew, dense, elev, denseref, locelev,         &
                        locdense, locconc, drhodc, crhoref, ctemp,             &
                        iform)
! ******************************************************************************
! buy_cf_riv -- Fill riv coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in), dimension(:) :: elev
    real(DP), intent(in) :: denseref
    integer(I4B), intent(in) :: locelev
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    integer(I4B), intent(in) :: iform
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: denseriv
    real(DP) :: elevriv
    real(DP) :: hriv
    real(DP) :: rbot
    real(DP) :: cond
    real(DP) :: hcofterm
    real(DP) :: rhsterm
! ------------------------------------------------------------------------------
    !
    ! -- Process density terms for each RIV
    do n = 1, packobj%nbound
      node = packobj%nodelist(n)
      if (packobj%ibound(node) <= 0) cycle
      !
      ! -- density
      denseriv = get_bnd_density(n, locdense, locconc, denseref, &
                                   drhodc, crhoref, ctemp, packobj%auxvar)
      !
      ! -- elevation
      elevriv = elev(node)
      if (locelev > 0) elevriv = packobj%auxvar(locelev, n)
      !
      ! -- boundary head and conductance
      hriv = packobj%bound(1, n)
      cond = packobj%bound(2, n)
      rbot = packobj%bound(3, n)
      !
      ! -- calculate and add terms depending on whether head is above rbot
      if (hnew(node) > rbot) then
        !
        ! --calculate HCOF and RHS terms, similar to GHB in this case
        call calc_ghb_hcof_rhs_terms(denseref, denseriv, dense(node),          &
                                     elevriv, elev(node), hriv, hnew(node),    &
                                     cond, iform, rhsterm, hcofterm)      
      else
        hcofterm = DZERO
        rhsterm = cond * (denseriv / denseref - DONE) * (hriv - rbot)
      end if
      !
      ! -- Add terms to package hcof and rhs accumulators
      packobj%hcof(n) = packobj%hcof(n) + hcofterm
      packobj%rhs(n) = packobj%rhs(n) - rhsterm
    end do
    !
    ! -- Return
    return
  end subroutine buy_cf_riv
  
  subroutine buy_cf_drn(packobj, hnew, dense, denseref)
! ******************************************************************************
! buy_cf_drn -- Fill drn coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in) :: denseref
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: rho
    real(DP) :: hbnd
    real(DP) :: cond
    real(DP) :: hcofterm
    real(DP) :: rhsterm
! ------------------------------------------------------------------------------
    !
    ! -- Process density terms for each DRN
    do n = 1, packobj%nbound
      node = packobj%nodelist(n)
      if (packobj%ibound(node) <= 0) cycle
      rho = dense(node)
      hbnd = packobj%bound(1, n)
      cond = packobj%bound(2, n)
      if (hnew(node) > hbnd) then
        hcofterm = - cond * (rho  / denseref - DONE)
        rhsterm = hcofterm * hbnd
        packobj%hcof(n) = packobj%hcof(n) + hcofterm
        packobj%rhs(n) = packobj%rhs(n) + rhsterm
      end if
    end do
    !
    ! -- Return
    return
  end subroutine buy_cf_drn
  
  subroutine buy_cf_lak(packobj, hnew, dense, elev, denseref, locdense,        &
                        locconc, drhodc, crhoref, ctemp, iform)
! ******************************************************************************
! buy_cf_lak -- Pass density information into lak package; density terms are
!   calculated in the lake package as part of lak_calculate_density_exchange
!   method
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    use LakModule, only: LakType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in), dimension(:) :: elev
    real(DP), intent(in) :: denseref
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    integer(I4B), intent(in) :: iform
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: denselak
! ------------------------------------------------------------------------------
    !
    ! -- Insert the lake and gwf relative densities into col 1 and 2 and the
    !    gwf elevation into col 3 of the lake package denseterms array
    select type(packobj)
    type is (LakType)
      do n = 1, packobj%nbound
        !
        ! -- get gwf node number
        node = packobj%nodelist(n)
        if (packobj%ibound(node) <= 0) cycle
        !
        ! -- Determine lak density
        denselak = get_bnd_density(n, locdense, locconc, denseref, &
                                   drhodc, crhoref, ctemp, packobj%auxvar)
        !
        ! -- fill lak relative density into column 1 of denseterms
        packobj%denseterms(1, n) = denselak / denseref
        !
        ! -- fill gwf relative density into column 2 of denseterms
        packobj%denseterms(2, n) = dense(node) / denseref
        !
        ! -- fill gwf elevation into column 3 of denseterms
        packobj%denseterms(3, n) = elev(node)
        !
      end do
    end select
    !
    ! -- Return
    return
  end subroutine buy_cf_lak
                        
  subroutine buy_cf_sfr(packobj, hnew, dense, elev, denseref, locdense,        &
                        locconc, drhodc, crhoref, ctemp, iform)
! ******************************************************************************
! buy_cf_sfr -- Pass density information into sfr package; density terms are
!   calculated in the sfr package as part of sfr_calculate_density_exchange
!   method
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    use SfrModule, only: SfrType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in), dimension(:) :: elev
    real(DP), intent(in) :: denseref
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    integer(I4B), intent(in) :: iform
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: densesfr
! ------------------------------------------------------------------------------
    !
    ! -- Insert the sfr and gwf relative densities into col 1 and 2 and the
    !    gwf elevation into col 3 of the sfr package denseterms array
    select type(packobj)
    type is (SfrType)
      do n = 1, packobj%nbound
        !
        ! -- get gwf node number
        node = packobj%nodelist(n)
        if (packobj%ibound(node) <= 0) cycle
        !
        ! -- Determine sfr density
        densesfr = get_bnd_density(n, locdense, locconc, denseref, &
                                   drhodc, crhoref, ctemp, packobj%auxvar)
        !
        ! -- fill sfr relative density into column 1 of denseterms
        packobj%denseterms(1, n) = densesfr / denseref
        !
        ! -- fill gwf relative density into column 2 of denseterms
        packobj%denseterms(2, n) = dense(node) / denseref
        !
        ! -- fill gwf elevation into column 3 of denseterms
        packobj%denseterms(3, n) = elev(node)
        !
      end do
    end select
    !
    ! -- Return
    return
  end subroutine buy_cf_sfr
                        
  subroutine buy_cf_maw(packobj, hnew, dense, elev, denseref, locdense,        &
                        locconc, drhodc, crhoref, ctemp, iform)
! ******************************************************************************
! buy_cf_maw -- Pass density information into maw package; density terms are
!   calculated in the maw package as part of maw_calculate_density_exchange
!   method
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use BndModule, only: BndType
    use MawModule, only: MawType
    class(BndType), pointer :: packobj
    ! -- dummy
    real(DP), intent(in), dimension(:) :: hnew
    real(DP), intent(in), dimension(:) :: dense
    real(DP), intent(in), dimension(:) :: elev
    real(DP), intent(in) :: denseref
    integer(I4B), intent(in) :: locdense
    integer(I4B), dimension(:), intent(in) :: locconc
    real(DP), dimension(:), intent(in) :: drhodc
    real(DP), dimension(:), intent(in) :: crhoref
    real(DP), dimension(:), intent(inout) :: ctemp
    integer(I4B), intent(in) :: iform
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: node
    real(DP) :: densemaw
! ------------------------------------------------------------------------------
    !
    ! -- Insert the maw and gwf relative densities into col 1 and 2 and the
    !    gwf elevation into col 3 of the maw package denseterms array
    select type(packobj)
    type is (MawType)
      do n = 1, packobj%nbound
        !
        ! -- get gwf node number
        node = packobj%nodelist(n)
        if (packobj%ibound(node) <= 0) cycle
        !
        ! -- Determine maw density
        densemaw = get_bnd_density(n, locdense, locconc, denseref, &
                                   drhodc, crhoref, ctemp, packobj%auxvar)
        !
        ! -- fill maw relative density into column 1 of denseterms
        packobj%denseterms(1, n) = densemaw / denseref
        !
        ! -- fill gwf relative density into column 2 of denseterms
        packobj%denseterms(2, n) = dense(node) / denseref
        !
        ! -- fill gwf elevation into column 3 of denseterms
        packobj%denseterms(3, n) = elev(node)
        !
      end do
    end select
    !
    ! -- Return
    return
  end subroutine buy_cf_maw
                        
  subroutine buy_fc(this, kiter, njasln, amat, idxglo, rhs, hnew)
! ******************************************************************************
! buy_fc -- Fill coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B) :: kiter
    integer,intent(in) :: njasln
    real(DP), dimension(njasln), intent(inout) :: amat
    integer(I4B), intent(in), dimension(:) :: idxglo
    real(DP), dimension(:), intent(inout) :: rhs
    real(DP), intent(inout), dimension(:) :: hnew
    ! -- local
    integer(I4B) :: n, m, ipos, idiag
    real(DP) :: rhsterm, amatnn, amatnm
! ------------------------------------------------------------------------------
    ! -- initialize
    amatnn = DZERO
    amatnm = DZERO
    !
    ! -- fill buoyancy flow term
    do n = 1, this%dis%nodes
      if(this%ibound(n) == 0) cycle
      idiag = this%dis%con%ia(n)
      do ipos = this%dis%con%ia(n) + 1, this%dis%con%ia(n + 1) - 1
        m = this%dis%con%ja(ipos)
        if(this%ibound(m) == 0) cycle
        if(this%iform == 0) then
          call this%calcbuy(n, m, ipos, hnew(n), hnew(m), rhsterm)
        else
          call this%calchhterms(n, m, ipos, hnew(n), hnew(m), rhsterm,         &
                                amatnn, amatnm)
        endif 
        !
        ! -- Add terms to rhs, diagonal, and off diagonal
        rhs(n) = rhs(n) - rhsterm
        amat(idxglo(idiag)) = amat(idxglo(idiag)) - amatnn
        amat(idxglo(ipos)) = amat(idxglo(ipos)) + amatnm
      enddo
    enddo
    !
    ! -- Return
    return
  end subroutine buy_fc

  subroutine buy_bdsav(this, idvfl, icbcfl, icbcun)
! ******************************************************************************
! buy_bdsav -- Fill coefficients
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B), intent(in) :: idvfl
    integer(I4B), intent(in) :: icbcfl
    integer(I4B), intent(in) :: icbcun
    ! -- local
    character(len=1) :: cdatafmp=' ', editdesc=' '
    integer(I4B) :: ibinun
    integer(I4B) :: iprint
    integer(I4B) :: nvaluesp
    integer(I4B) :: nwidthp
    real(DP) :: dinact
! ------------------------------------------------------------------------------
    !
    ! -- Set unit number for density output
    if(this%ioutdense /= 0) then
      ibinun = 1
    else
      ibinun = 0
    endif
    if(idvfl == 0) ibinun = 0
    !
    ! -- save density array
    if (ibinun /= 0) then
      iprint = 0
      dinact = DHNOFLO
      !
      ! -- write density to binary file
      if (this%ioutdense /= 0) then
        ibinun = this%ioutdense
        call this%dis%record_array(this%dense, this%iout, iprint, ibinun,       &
                                   '         DENSITY', cdatafmp, nvaluesp,      &
                                   nwidthp, editdesc, dinact)
      end if
    end if
    
    !
    ! -- Return
    return
  end subroutine buy_bdsav

  subroutine buy_flowja(this, hnew, flowja)
! ******************************************************************************
! buy_flowja -- Add buy term to flowja
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    implicit none
    class(GwfBuyType) :: this
    real(DP),intent(in),dimension(:) :: hnew
    real(DP),intent(inout),dimension(:) :: flowja
    integer(I4B) :: n, m, ipos
    real(DP) :: deltaQ
    real(DP) :: rhsterm, amatnn, amatnm
! ------------------------------------------------------------------------------
    !
    ! -- Calculate the flow across each cell face and store in flowja
    do n=1,this%dis%nodes
      do ipos = this%dis%con%ia(n) + 1, this%dis%con%ia(n + 1) - 1
        m = this%dis%con%ja(ipos)
        if(m < n) cycle
        if(this%iform == 0) then
          ! -- equivalent freshwater head formulation
          call this%calcbuy(n, m, ipos, hnew(n), hnew(m), deltaQ)
        else
          ! -- hydraulic head formulation
          call this%calchhterms(n, m, ipos, hnew(n), hnew(m), rhsterm,         &
                                amatnn, amatnm)
          deltaQ = amatnm * hnew(m) - amatnn * hnew(n) + rhsterm
        endif
        flowja(ipos) = flowja(ipos) + deltaQ
        flowja(this%dis%con%isym(ipos)) = flowja(this%dis%con%isym(ipos)) -    &
          deltaQ
      enddo
    enddo
    !
    ! -- Return
    return
  end subroutine buy_flowja
 
  subroutine buy_da(this)
! ******************************************************************************
! buy_da -- Deallocate
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
! ------------------------------------------------------------------------------
    !
    ! -- Deallocate arrays if package was active
    if(this%inunit > 0) then
      call mem_deallocate(this%elev)
      call mem_deallocate(this%dense)
      call mem_deallocate(this%concbuy)
      call mem_deallocate(this%drhodc)
      call mem_deallocate(this%crhoref)
      call mem_deallocate(this%ctemp)
      deallocate(this%cmodelname)
      deallocate(this%cauxspeciesname)
      deallocate(this%modelconc)
    endif
    !
    ! -- Scalars
    call mem_deallocate(this%ioutdense)
    call mem_deallocate(this%iform)
    call mem_deallocate(this%ireadelev)
    call mem_deallocate(this%ireadconcbuy)
    call mem_deallocate(this%iconcset)
    call mem_deallocate(this%denseref)
    
    call mem_deallocate(this%nrhospecies)
    !
    ! -- deallocate parent
    call this%NumericalPackageType%da()
    !
    ! -- Return
    return
  end subroutine buy_da

  subroutine read_dimensions(this)
! ******************************************************************************
! read_dimensions -- Read the dimensions for this package
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType),intent(inout) :: this
    ! -- local
    character (len=LINELENGTH) :: errmsg, keyword
    integer(I4B) :: ierr
    logical :: isfound, endOfBlock
    ! -- format
! ------------------------------------------------------------------------------
    !
    ! -- get dimensions block
    call this%parser%GetBlock('DIMENSIONS', isfound, ierr, &
                              supportOpenClose=.true.)
    !
    ! -- parse dimensions block if detected
    if (isfound) then
      write(this%iout,'(/1x,a)')'PROCESSING BUY DIMENSIONS'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case ('NRHOSPECIES')
            this%nrhospecies = this%parser%GetInteger()
            write(this%iout,'(4x,a,i0)')'NRHOSPECIES = ', this%nrhospecies
          case default
            write(errmsg,'(4x,a,a)')                                           &
              'UNKNOWN BUY DIMENSION: ', trim(keyword)
            call store_error(errmsg)
            call this%parser%StoreErrorUnit()
            call ustop()
        end select
      end do
      write(this%iout,'(1x,a)')'END OF BUY DIMENSIONS'
    else
      call store_error('REQUIRED BUY DIMENSIONS BLOCK NOT FOUND.')
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- check dimension
    if (this%nrhospecies < 1) then
      call store_error('NRHOSPECIES MUST BE GREATER THAN ONE.')
      call this%parser%StoreErrorUnit()
      call ustop()
    endif
    !
    ! -- return
    return
  end subroutine read_dimensions

  subroutine read_packagedata(this)
! ******************************************************************************
! read_packagedata -- Read PACKAGEDATA block
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg
    character(len=LINELENGTH) :: line
    integer(I4B) :: ierr
    integer(I4B) :: irhospec
    logical :: isfound, endOfBlock
    logical :: blockrequired
    integer(I4B), dimension(:), allocatable :: itemp
    character(len=10) :: c10
    character(len=16) :: c16
    ! -- format
    character(len=*),parameter :: fmterr = &
      "('INVALID VALUE FOR IRHOSPEC (',i0,') DETECTED IN BUY PACKAGE. &
      &IRHOSPEC MUST BE > 0 AND < NRHOSPECIES, AND DUPLICATE VALUES &
      &ARE NOT ALLOWED.')"
! ------------------------------------------------------------------------------
    !
    ! -- initialize
    allocate(itemp(this%nrhospecies))
    itemp(:) = 0
    !
    ! -- get packagedata block
    blockrequired = .true.
    call this%parser%GetBlock('PACKAGEDATA', isfound, ierr,                    &
                              blockRequired=blockRequired,                     &
                              supportOpenClose=.true.)
    !
    ! -- parse packagedata block
    if (isfound) then
      write(this%iout,'(1x,a)')'PROCESSING BUY PACKAGEDATA'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        irhospec = this%parser%GetInteger()
        if (irhospec < 1 .or. irhospec > this%nrhospecies) then
          write(errmsg, fmterr) irhospec
          call store_error(errmsg)
        end if
        if (itemp(irhospec) /= 0) then
          write(errmsg, fmterr) irhospec
          call store_error(errmsg)
        end if
        itemp(irhospec) = 1
        this%drhodc(irhospec) = this%parser%GetDouble()
        this%crhoref(irhospec) = this%parser%GetDouble()
        call this%parser%GetStringCaps(this%cmodelname(irhospec))
        call this%parser%GetStringCaps(this%cauxspeciesname(irhospec))
      end do
      write(this%iout,'(1x,a)') 'END OF BUY PACKAGEDATA'
    end if
    !
    ! -- Check for errors.
    if (count_errors() > 0) then
      call this%parser%StoreErrorUnit()
      call ustop()
    end if
    !
    ! -- write packagedata information
    write(this%iout, '(/,a)') 'SUMMARY OF SPECIES INFORMATION IN BUY PACKAGE'
    write(this%iout, '(1a11, 4a17)') &
                         'SPECIES', 'DRHODC', 'CRHOREF', 'MODEL', &
                         'AUXSPECIESNAME'
    do irhospec = 1, this%nrhospecies
      write(c10, '(i0)') irhospec
      line = ' ' // adjustr(c10)
      write(c16, '(g15.6)') this%drhodc(irhospec)
      line = trim(line) // ' ' // adjustr(c16)
      write(c16, '(g15.6)') this%crhoref(irhospec)
      line = trim(line) // ' ' // adjustr(c16)
      write(c16, '(a)') this%cmodelname(irhospec)
      line = trim(line) // ' ' // adjustr(c16)
      write(c16, '(a)') this%cauxspeciesname(irhospec)
      line = trim(line) // ' ' // adjustr(c16)
      write(this%iout, '(a)') trim(line)
    end do
    !
    ! -- deallocate
    deallocate(itemp)
    !
    ! -- return
    return
  end subroutine read_packagedata
  
  subroutine calcbuy(this, n, m, icon, hn, hm, buy)
! ******************************************************************************
! calcbuy -- Calculate buyancy term for this connection
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use GwfNpfModule, only: hcond, vcond
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B), intent(in) :: n
    integer(I4B), intent(in) :: m
    integer(I4B), intent(in) :: icon
    real(DP), intent(in) :: hn
    real(DP), intent(in) :: hm
    real(DP), intent(inout) :: buy
    ! -- local
    integer(I4B) :: ihc
    real(DP) :: densen, densem, cl1, cl2, avgdense, wt, elevn, elevm, &
                     cond, tp, bt
    real(DP) :: hyn
    real(DP) :: hym
! ------------------------------------------------------------------------------
    !
    ! -- Average density
    densen = this%dense(n)
    densem = this%dense(m)
    if (m > n) then
      cl1 = this%dis%con%cl1(this%dis%con%jas(icon))
      cl2 = this%dis%con%cl2(this%dis%con%jas(icon))
    else
      cl1 = this%dis%con%cl2(this%dis%con%jas(icon))
      cl2 = this%dis%con%cl1(this%dis%con%jas(icon))
    end if
    wt =  cl1 / (cl1 + cl2)
    avgdense = wt * densen + (DONE - wt) * densem
    !
    ! -- Elevations
    if(this%ireadelev == 0) then
      tp = this%dis%top(n)
      bt = this%dis%bot(n)
      elevn = bt + DHALF * this%npf%sat(n) * (tp - bt)
      tp = this%dis%top(m)
      bt = this%dis%bot(m)
      elevm = bt + DHALF * this%npf%sat(m) * (tp - bt)
    else
      elevn = this%elev(n)
      elevm = this%elev(m)
    endif
    !
    ihc = this%dis%con%ihc(this%dis%con%jas(icon))
    hyn = this%npf%hy_eff(n, m, ihc, ipos=icon)
    hym = this%npf%hy_eff(m, n, ihc, ipos=icon)
    !
    ! -- Conductance
    if(this%dis%con%ihc(this%dis%con%jas(icon)) == 0) then
      cond = vcond(this%ibound(n), this%ibound(m),                             &
                      this%npf%icelltype(n), this%npf%icelltype(m),            &
                      this%npf%inewton,                                        &
                      this%npf%ivarcv, this%npf%idewatcv,                      &
                      this%npf%condsat(this%dis%con%jas(icon)), hn, hm,        &
                      hyn, hym,                                                &
                      this%npf%sat(n), this%npf%sat(m),                        &
                      this%dis%top(n), this%dis%top(m),                        &
                      this%dis%bot(n), this%dis%bot(m),                        &
                      this%dis%con%hwva(this%dis%con%jas(icon)))
    else
      cond = hcond(this%ibound(n), this%ibound(m),                             &
                     this%npf%icelltype(n), this%npf%icelltype(m),             &
                     this%npf%inewton, this%npf%inewton,                       &
                     this%dis%con%ihc(this%dis%con%jas(icon)),                 &
                     this%npf%icellavg, this%npf%iusgnrhc, this%npf%inwtupw,   &
                     this%npf%condsat(this%dis%con%jas(icon)),                 &
                     hn, hm, this%npf%sat(n), this%npf%sat(m),                 &
                     hyn, hym,                                                 &
                     this%dis%top(n), this%dis%top(m),                         &
                     this%dis%bot(n), this%dis%bot(m),                         &
                     this%dis%con%cl1(this%dis%con%jas(icon)),                 &
                     this%dis%con%cl2(this%dis%con%jas(icon)),                 &
                     this%dis%con%hwva(this%dis%con%jas(icon)),                &
                     this%npf%satomega, this%npf%satmin)
    endif
    !
    ! -- Calculate buoyancy term
    buy = cond * (avgdense - this%denseref) / this%denseref * (elevm - elevn)
    !
    ! -- Return
    return
  end subroutine calcbuy


  subroutine calchhterms(this, n, m, icon, hn, hm, rhsterm, amatnn, amatnm)
! ******************************************************************************
! calchhterms -- Calculate hydraulic head term for this connection
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use GwfNpfModule, only: hcond, vcond
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B), intent(in) :: n
    integer(I4B), intent(in) :: m
    integer(I4B), intent(in) :: icon
    real(DP), intent(in) :: hn
    real(DP), intent(in) :: hm
    real(DP), intent(inout) :: rhsterm
    real(DP), intent(inout) :: amatnn
    real(DP), intent(inout) :: amatnm
    ! -- local
    integer(I4B) :: ihc
    real(DP) :: densen, densem, cl1, cl2, avgdense, wt, elevn, elevm, cond
    real(DP) :: rhonormn, rhonormm
    real(DP) :: rhoterm
    real(DP) :: elevnm
    real(DP) :: hphi
    real(DP) :: hyn
    real(DP) :: hym
! ------------------------------------------------------------------------------
    !
    ! -- Average density
    densen = this%dense(n)
    densem = this%dense(m)
    if (m > n) then
      cl1 = this%dis%con%cl1(this%dis%con%jas(icon))
      cl2 = this%dis%con%cl2(this%dis%con%jas(icon))
    else
      cl1 = this%dis%con%cl2(this%dis%con%jas(icon))
      cl2 = this%dis%con%cl1(this%dis%con%jas(icon))
    end if
    wt =  cl1 / (cl1 + cl2)
    avgdense = wt * densen + (1.0 - wt) * densem
    !
    ! -- Elevations
    elevn = this%elev(n)
    elevm = this%elev(m)
    elevnm = (DONE - wt) * elevn + wt * elevm
    !
    ihc = this%dis%con%ihc(this%dis%con%jas(icon))
    hyn = this%npf%hy_eff(n, m, ihc, ipos=icon)
    hym = this%npf%hy_eff(m, n, ihc, ipos=icon)
    !
    ! -- Conductance
    if(ihc == 0) then
      cond = vcond(this%ibound(n), this%ibound(m),                             &
                      this%npf%icelltype(n), this%npf%icelltype(m),            &
                      this%npf%inewton,                                        &
                      this%npf%ivarcv, this%npf%idewatcv,                      &
                      this%npf%condsat(this%dis%con%jas(icon)), hn, hm,        &
                      hyn, hym,                                                &
                      this%npf%sat(n), this%npf%sat(m),                        &
                      this%dis%top(n), this%dis%top(m),                        &
                      this%dis%bot(n), this%dis%bot(m),                        &
                      this%dis%con%hwva(this%dis%con%jas(icon)))
    else
      cond = hcond(this%ibound(n), this%ibound(m),                             &
                     this%npf%icelltype(n), this%npf%icelltype(m),             &
                     this%npf%inewton, this%npf%inewton,                       &
                     this%dis%con%ihc(this%dis%con%jas(icon)),                 &
                     this%npf%icellavg, this%npf%iusgnrhc, this%npf%inwtupw,   &
                     this%npf%condsat(this%dis%con%jas(icon)),                 &
                     hn, hm, this%npf%sat(n), this%npf%sat(m),                 &
                     hyn, hym,                                                 &
                     this%dis%top(n), this%dis%top(m),                         &
                     this%dis%bot(n), this%dis%bot(m),                         &
                     this%dis%con%cl1(this%dis%con%jas(icon)),                 &
                     this%dis%con%cl2(this%dis%con%jas(icon)),                 &
                     this%dis%con%hwva(this%dis%con%jas(icon)),                &
                     this%npf%satomega, this%npf%satmin)
    endif
    !
    ! -- Calculate terms
    rhonormn = densen / this%denseref
    rhonormm = densem / this%denseref
    rhoterm = wt * rhonormn + (DONE - wt) * rhonormm
    amatnn = cond * (rhoterm - DONE)
    amatnm = amatnn
    rhsterm = -cond * (rhonormm - rhonormn) * elevnm
    if (this%iform == 1) then
      ! -- rhs (lag the h terms and keep matrix symmetric)
      hphi = (DONE - wt) * hn + wt * hm
      rhsterm = rhsterm + cond * hphi * (rhonormm - rhonormn)
    else
      ! -- lhs, results in asymmetric matrix due to weight term
      amatnn = amatnn - cond * (DONE - wt) * (rhonormm - rhonormn)
      amatnm = amatnm + cond * wt * (rhonormm - rhonormn)
    endif
    !
    ! -- Return
    return
  end subroutine calchhterms

  subroutine buy_calcdens(this)
! ******************************************************************************
! buy_calcdens -- calculate fluid density from concentration
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    
    ! -- local
    integer(I4B) :: n
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- Calculate the density using the specified concentration array
    do n = 1, this%dis%nodes
      do i = 1, this%nrhospecies
        if(this%modelconc(i)%icbund(n) == 0) then
          this%ctemp = DZERO
        else
          this%ctemp(i) = this%modelconc(i)%conc(n)
        end if
      end do
      this%dense(n) = calcdens(this%denseref, this%drhodc, this%crhoref, &
                               this%ctemp)
    enddo
    !
    ! -- Return
    return
  end subroutine buy_calcdens
  
  subroutine buy_calcelev(this)
! ******************************************************************************
! buy_calcelev -- Calculate cell elevations to use in density flow equations
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
    integer(I4B) :: n
    real(DP) :: tp, bt, frac
! ------------------------------------------------------------------------------
    !
    ! -- Calculate the elev array
    do n = 1, this%dis%nodes
      tp = this%dis%top(n)
      bt = this%dis%bot(n)
      frac = this%npf%sat(n)
      this%elev(n) = bt + DHALF * frac * (tp - bt)
    end do
    !
    ! -- Return
    return
  end subroutine buy_calcelev
  
  subroutine allocate_scalars(this)
! ******************************************************************************
! allocate_scalars
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use ConstantsModule, only: DZERO
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
! ------------------------------------------------------------------------------
    !
    ! -- allocate scalars in NumericalPackageType
    call this%NumericalPackageType%allocate_scalars()
    !
    ! -- Allocate
    call mem_allocate(this%ioutdense, 'IOUTDENSE', this%memoryPath)
    call mem_allocate(this%iform, 'IFORM', this%memoryPath)
    call mem_allocate(this%ireadelev, 'IREADELEV', this%memoryPath)
    call mem_allocate(this%ireadconcbuy, 'IREADCONCBUY', this%memoryPath)
    call mem_allocate(this%iconcset, 'ICONCSET', this%memoryPath)
    call mem_allocate(this%denseref, 'DENSEREF', this%memoryPath)

    call mem_allocate(this%nrhospecies, 'NRHOSPECIES', this%memoryPath)
    
    !
    ! -- Initialize
    this%ioutdense = 0
    this%ireadelev = 0
    this%iconcset = 0
    this%ireadconcbuy = 0
    this%denseref = 1000.d0
    
    this%nrhospecies = 0
    
    !
    ! -- Initialize default to LHS implementation of hydraulic head formulation
    this%iform = 2
    this%iasym = 1
    !
    ! -- Return
    return
  end subroutine allocate_scalars

  subroutine allocate_arrays(this, nodes)
! ******************************************************************************
! allocate_arrays
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
    integer(I4B), intent(in) :: nodes
    ! -- local
    integer(I4B) :: i
! ------------------------------------------------------------------------------
    !
    ! -- Allocate
    call mem_allocate(this%dense, nodes, 'DENSE', this%memoryPath)
    call mem_allocate(this%concbuy, 0, 'CONCBUY', this%memoryPath)
    call mem_allocate(this%elev, nodes, 'ELEV', this%memoryPath)
    call mem_allocate(this%drhodc, this%nrhospecies, 'DRHODC', this%memoryPath)
    call mem_allocate(this%crhoref, this%nrhospecies, 'CRHOREF', this%memoryPath)
    call mem_allocate(this%ctemp, this%nrhospecies, 'CTEMP', this%memoryPath)
    allocate(this%cmodelname(this%nrhospecies))
    allocate(this%cauxspeciesname(this%nrhospecies))
    allocate(this%modelconc(this%nrhospecies))
    !
    ! -- Initialize
    do i = 1, nodes
      this%dense(i) = this%denseref
      this%elev(i) = DZERO
    enddo
    !
    ! -- Initialize nrhospecies arrays
    do i = 1, this%nrhospecies
      this%drhodc(i) = DZERO
      this%crhoref(i) = DZERO
      this%ctemp(i) = DZERO
      this%cmodelname(i) = ''
      this%cauxspeciesname(i) = ''
    end do
    !
    ! -- Return
    return
  end subroutine allocate_arrays

  subroutine read_options(this)
! ******************************************************************************
! read_options
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    use OpenSpecModule, only: access, form
    use InputOutputModule, only: urword, getunit, urdaux, openfile
    ! -- dummy
    class(GwfBuyType) :: this
    ! -- local
    character(len=LINELENGTH) :: errmsg, keyword
    character(len=MAXCHARLEN) :: fname
    integer(I4B) :: ierr
    logical :: isfound, endOfBlock
    ! -- formats
    character(len=*),parameter :: fmtfileout = &
      "(4x, 'BUY ', 1x, a, 1x, ' WILL BE SAVED TO FILE: ', a, /4x, 'OPENED ON UNIT: ', I7)"
! ------------------------------------------------------------------------------
    !
    ! -- get options block
    call this%parser%GetBlock('OPTIONS', isfound, ierr, &
                              supportOpenClose=.true., blockRequired=.false.)
    !
    ! -- parse options block if detected
    if (isfound) then
      write(this%iout,'(1x,a)')'PROCESSING BUY OPTIONS'
      do
        call this%parser%GetNextLine(endOfBlock)
        if (endOfBlock) exit
        call this%parser%GetStringCaps(keyword)
        select case (keyword)
          case ('HHFORMULATION_RHS')
            this%iform = 1
            this%iasym = 0
            write(this%iout,'(4x,a)') &
              'HYDDRAULIC HEAD FORMULATION SET TO RIGHT-HAND SIDE'
          case ('DENSEREF')
            this%denseref = this%parser%GetDouble()
            write(this%iout, '(4x,a,1pg15.6)')                                 &
                             'REFERENCE DENSITY HAS BEEN SET TO: ',  &
                             this%denseref
          case ('DEV_EFH_FORMULATION')
            call this%parser%DevOpt()
            this%iform = 0
            this%iasym = 0
            write(this%iout,'(4x,a)') &
              'FORMULATION SET TO EQUIVALENT FRESHWATER HEAD'
          case ('DENSITY')
            call this%parser%GetStringCaps(keyword)
            if (keyword == 'FILEOUT') then
              call this%parser%GetString(fname)
              this%ioutdense = getunit()
              call openfile(this%ioutdense, this%iout, fname, 'DATA(BINARY)',  &
                            form, access, 'REPLACE')
              write(this%iout,fmtfileout) &
                'DENSITY', fname, this%ioutdense
            else 
              errmsg = 'OPTIONAL DENSITY KEYWORD MUST BE ' //                  &
                       'FOLLOWED BY FILEOUT'
              call store_error(errmsg)
            end if
          case default
            write(errmsg,'(4x,a,a)')'****ERROR. UNKNOWN BUY OPTION: ',         &
                                     trim(keyword)
            call store_error(errmsg)
            call this%parser%StoreErrorUnit()
            call ustop()
        end select
      end do
      write(this%iout,'(1x,a)')'END OF BUY OPTIONS'
    end if
    !
    ! -- Return
    return
  end subroutine read_options

  subroutine set_concentration_pointer(this, modelname, conc, icbund)
! ******************************************************************************
! set_concentration_pointer -- pass in a gwt model name, concentration array
!   and ibound, and store a pointer to these in the BUY package so that 
!   density can be calculated from them.
!   This routine is called from the gwfgwt exchange in the exg_ar() method.
! ******************************************************************************
!
!    SPECIFICATIONS:
! ------------------------------------------------------------------------------
    ! -- modules
    ! -- dummy
    class(GwfBuyType) :: this
    character(len=LENMODELNAME), intent(in) :: modelname
    real(DP), dimension(:), pointer :: conc
    integer(I4B), dimension(:), pointer :: icbund
    ! -- local
    integer(I4B) :: i
    logical :: found
! ------------------------------------------------------------------------------
    !
    this%iconcset = 1
    found = .false.
    do i = 1, this%nrhospecies
      if (this%cmodelname(i) == modelname) then
        this%modelconc(i)%conc => conc
        this%modelconc(i)%icbund => icbund
        found = .true.
        exit
      end if
    end do
    !
    ! -- Return
    return
  end subroutine set_concentration_pointer
  
end module GwfBuyModule